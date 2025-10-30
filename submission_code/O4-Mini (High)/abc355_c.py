import sys
import threading

def main():
    data = sys.stdin.read().split()
    if not data:
        print(-1)
        return

    it = iter(data)
    N = int(next(it))
    T = int(next(it))

    # Counters for rows, columns, and the two diagonals
    rows = [0] * N
    cols = [0] * N
    diag = 0
    anti = 0

    answer = -1
    for turn in range(1, T + 1):
        try:
            a = int(next(it))
        except StopIteration:
            break
        # Compute zero-based row i and column j
        a0 = a - 1
        i = a0 // N
        j = a0 % N

        # Mark the row and column
        rows[i] += 1
        if rows[i] == N:
            answer = turn
            break

        cols[j] += 1
        if cols[j] == N:
            answer = turn
            break

        # Mark main diagonal if applicable
        if i == j:
            diag += 1
            if diag == N:
                answer = turn
                break

        # Mark anti-diagonal if applicable
        if i + j == N - 1:
            anti += 1
            if anti == N:
                answer = turn
                break

    # Print the result
    print(answer if answer != -1 else -1)

if __name__ == "__main__":
    main()