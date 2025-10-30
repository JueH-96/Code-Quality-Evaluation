import sys

def calculate_fatigue(n, presses):
    """
    Calculate the minimum possible fatigue level at the end of the performance.

    Args:
    n (int): The number of key presses.
    presses (list): A list of tuples, where each tuple contains the key and hand used for each press.

    Returns:
    int: The minimum possible fatigue level.
    """
    # Initialize the fatigue level and the positions of the hands
    fatigue = 0
    left_hand = presses[0][0] if presses[0][1] == 'L' else presses[1][0]
    right_hand = presses[0][0] if presses[0][1] == 'R' else presses[1][0]

    # Iterate over the presses
    for key, hand in presses:
        # If the hand is not on the key, move it and increase the fatigue level
        if hand == 'L' and left_hand != key:
            fatigue += abs(key - left_hand)
            left_hand = key
        elif hand == 'R' and right_hand != key:
            fatigue += abs(key - right_hand)
            right_hand = key

    return fatigue

def main():
    # Read the input
    n = int(sys.stdin.readline())
    presses = []
    for _ in range(n):
        key, hand = sys.stdin.readline().split()
        presses.append((int(key), hand))

    # Calculate and print the minimum possible fatigue level
    print(calculate_fatigue(n, presses))

if __name__ == "__main__":
    main()