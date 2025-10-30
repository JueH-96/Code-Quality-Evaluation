def calculate_fatigue(N, presses):
    # Initialize the positions of both hands to None
    left_hand, right_hand = None, None
    fatigue = 0

    for key, hand in presses:
        if hand == 'L':
            if left_hand is not None:
                fatigue += abs(key - left_hand)
            left_hand = key
        else:
            if right_hand is not None:
                fatigue += abs(key - right_hand)
            right_hand = key

    return fatigue

# Read the number of keys to be pressed
N = int(input().strip())

# Read the sequence of key presses
presses = [tuple(input().strip().split()) for _ in range(N)]
presses = [(int(key), hand) for key, hand in presses]

# Calculate and print the minimum fatigue level
print(calculate_fatigue(N, presses))