import sys

def main() -> None:
    input_data = sys.stdin.read().strip().splitlines()
    N = int(input_data[0])
    S = input_data[1].strip()

    # prefix_ones[i] : length of consecutive '1's ending at position i
    prefix_ones = [0] * N
    for i in range(N):
        if S[i] == '1':
            prefix_ones[i] = prefix_ones[i - 1] + 1 if i else 1
        else:
            prefix_ones[i] = 0

    # suffix_twos[i] : length of consecutive '2's starting at position i
    suffix_twos = [0] * N
    for i in range(N - 1, -1, -1):
        if S[i] == '2':
            suffix_twos[i] = suffix_twos[i + 1] + 1 if i + 1 < N else 1
        else:
            suffix_twos[i] = 0

    best = 1      # at least one '/' exists, so answer ≥ 1
    for i, ch in enumerate(S):
        if ch == '/':
            left_ones  = prefix_ones[i - 1] if i > 0 else 0
            right_twos = suffix_twos[i + 1] if i + 1 < N else 0
            k = min(left_ones, right_twos)
            best = max(best, 2 * k + 1)

    print(best)

if __name__ == "__main__":
    main()