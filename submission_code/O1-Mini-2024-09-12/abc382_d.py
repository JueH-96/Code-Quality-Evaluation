import sys

def main():
    import sys

    N, M = map(int, sys.stdin.read().split())
    sequences = []

    def backtrack(current_sequence, position):
        if position == N:
            if current_sequence[-1] <= M:
                sequences.append(list(current_sequence))
            return
        else:
            prev = current_sequence[-1]
            min_next = prev + 10
            # Calculate the maximum possible next value to ensure A_N <= M
            max_next = M - 10 * (N - position - 1)
            for next_val in range(min_next, max_next + 1):
                backtrack(current_sequence + [next_val], position + 1)

    # The first element can range from 1 to M - 10*(N-1)
    min_start = 1
    max_start = M - 10 * (N - 1)
    for first in range(min_start, max_start + 1):
        backtrack([first], 1)

    print(len(sequences))
    for seq in sequences:
        print(' '.join(map(str, seq)))

if __name__ == "__main__":
    main()