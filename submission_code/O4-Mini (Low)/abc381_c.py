import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1].strip()

    # Precompute for each position i the number of consecutive '2's starting at i
    next_twos = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        if S[i] == '2':
            next_twos[i] = next_twos[i + 1] + 1
        else:
            next_twos[i] = 0

    max_len = 1
    left_ones = 0  # count of consecutive '1's ending at previous position

    for i, ch in enumerate(S):
        if ch == '1':
            left_ones += 1
        else:
            # reset count if not '1'
            left_ones = 0

        if ch == '/':
            # For slash at i, left_ones is count of '1's to its left
            # right_twos is count of '2's to its right
            right_twos = next_twos[i + 1] if i + 1 < N else 0
            k = min(left_ones, right_twos)
            length = 2 * k + 1
            if length > max_len:
                max_len = length

    print(max_len)

if __name__ == "__main__":
    main()