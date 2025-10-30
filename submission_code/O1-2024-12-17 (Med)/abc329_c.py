def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # Array to keep track of the maximum consecutive run for each character
    max_run = [0] * 26

    current_char = S[0]
    current_run = 1

    for i in range(1, N):
        if S[i] == current_char:
            current_run += 1
        else:
            # Update the maximum run length for the previous character
            idx = ord(current_char) - ord('a')
            max_run[idx] = max(max_run[idx], current_run)

            # Reset for the new character
            current_char = S[i]
            current_run = 1

    # Don't forget the last segment
    idx = ord(current_char) - ord('a')
    max_run[idx] = max(max_run[idx], current_run)

    # The result is the sum of maximum consecutive run lengths across all characters
    print(sum(max_run))

# Do not forget to call main()
if __name__ == "__main__":
    main()