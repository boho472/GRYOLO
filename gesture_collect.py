# gesture data collection script for Transformer model training
# saves hand landmark sequences instead of photos
# each sample is 30 frames of hand coordinates recorded over 1 second

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# list of gestures we want to collect data for
GESTURES = ["wave", "fist", "point", "thumbs_up", "open_palm"]

# how many recordings per gesture, needs to be 300 for Transformer to train well
SAMPLES_PER_GESTURE = 300

# how many frames per recording, 30 frames is roughly 1 second of motion
FRAMES_PER_SAMPLE = 30

# folder where all gesture data files will be saved
SAVE_DIR = "gesture_data"

os.makedirs(SAVE_DIR, exist_ok=True)

# set up mediapipe to detect one hand at a time
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def extract_landmarks(hand_landmarks):
    # pulls x and y coordinates from all 21 hand landmarks
    # gives us 42 numbers per frame representing the hand position
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x)
        coords.append(lm.y)
    return np.array(coords)

def collect_gesture(gesture_name):
    # collects 300 samples for one gesture
    # each sample is 30 frames recorded when you press space
    print(f"\n{'='*50}")
    print(f"  Gesture: {gesture_name.upper()}")
    print(f"  Need: {SAMPLES_PER_GESTURE} samples x {FRAMES_PER_SAMPLE} frames each")
    print(f"{'='*50}")
    print("Press SPACE to record each sample. Press Q to quit.\n")

    # if we already collected some samples before, resume from where we left off
    save_path = os.path.join(SAVE_DIR, f"{gesture_name}.npy")
    if os.path.exists(save_path):
        existing = np.load(save_path)
        all_samples = list(existing)
        print(f"  Resuming from {len(all_samples)} samples already collected")
    else:
        all_samples = []

    cap = cv2.VideoCapture(0)
    sample_count = len(all_samples)

    while sample_count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            break

        # flip frame so it feels like a mirror
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # draw hand landmarks on screen so user can see detection
        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        # show current gesture name and progress on screen
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {sample_count}/{SAMPLES_PER_GESTURE}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE = record | Q = quit", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("Gesture Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord(' '):
            # when space is pressed, record 30 frames of the gesture
            sequence = []
            success = True

            for frame_num in range(FRAMES_PER_SAMPLE):
                ret, frame = cap.read()
                if not ret:
                    success = False
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    # extract the 42 landmark coordinates for this frame
                    landmarks = extract_landmarks(result.multi_hand_landmarks[0])
                    sequence.append(landmarks)
                    for hand_lm in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                else:
                    # if no hand detected in this frame, fill with zeros
                    sequence.append(np.zeros(42))

                # show recording status on screen
                cv2.putText(frame, f"RECORDING... {frame_num+1}/{FRAMES_PER_SAMPLE}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Gesture Collection", frame)
                cv2.waitKey(1)

            # only save the sample if all 30 frames were recorded successfully
            if success and len(sequence) == FRAMES_PER_SAMPLE:
                all_samples.append(np.array(sequence))
                sample_count += 1
                print(f"  Sample {sample_count}/{SAMPLES_PER_GESTURE} saved")

                # save progress to disk every 10 samples so we dont lose data
                if sample_count % 10 == 0:
                    np.save(save_path, np.array(all_samples))
                    print(f"  Progress saved ({sample_count} samples)")

    cap.release()
    cv2.destroyAllWindows()

    # save all collected samples to a npy file when done
    if all_samples:
        data = np.array(all_samples)
        np.save(save_path, data)
        print(f"\nDone. {len(all_samples)} samples saved to {save_path}")
        print(f"Data shape: {data.shape}  (samples x frames x landmarks)")

def main():
    print("\n" + "="*50)
    print("   GESTURE DATA COLLECTION — TRANSFORMER VERSION")
    print("="*50)
    print(f"Gestures: {GESTURES}")
    print(f"Samples per gesture: {SAMPLES_PER_GESTURE}")
    print(f"Frames per sample:   {FRAMES_PER_SAMPLE}")

    # let user pick which gesture to collect or collect all at once
    print("\nWhich gesture?")
    for i, g in enumerate(GESTURES):
        print(f"  {i+1}. {g}")
    print(f"  {len(GESTURES)+1}. ALL gestures")

    choice = input("\nEnter number: ").strip()

    if choice == str(len(GESTURES)+1):
        # collect all gestures one by one with a short break in between
        for gesture in GESTURES:
            collect_gesture(gesture)
            print("\nRest for 5 seconds...")
            time.sleep(5)
    elif choice.isdigit() and 1 <= int(choice) <= len(GESTURES):
        collect_gesture(GESTURES[int(choice)-1])
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()