def main():
    import sys

    # Read the keyboard arrangement string S
    S = sys.stdin.readline().strip()

    # Create a mapping from character to its coordinate (1-based index)
    pos = {ch: i for i, ch in enumerate(S, start=1)}

    # Define the sequence of characters we must type
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Sum the absolute distances between consecutive keys in the typing sequence
    total_distance = 0
    for prev_char, curr_char in zip(alphabet, alphabet[1:]):
        total_distance += abs(pos[curr_char] - pos[prev_char])

    # Output the result
    print(total_distance)

if __name__ == "__main__":
    main()