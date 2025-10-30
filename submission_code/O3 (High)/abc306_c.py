import sys

def main() -> None:
    # Read all integers at once for speed.
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:               # Empty input guard (shouldn't occur in valid tests)
        return

    N = data[0]
    A = data[1:]               # Sequence of length 3N

    # counts[i]   : how many times we've seen value i so far
    # middle[i]   : position (1-based) of the 2nd (middle) occurrence of value i
    counts = [0] * (N + 1)
    middle = [0] * (N + 1)

    for idx, v in enumerate(A, start=1):  # idx is 1-based position
        counts[v] += 1
        if counts[v] == 2:                # middle occurrence
            middle[v] = idx

    # Sort 1..N by their recorded middle position
    order = sorted(range(1, N + 1), key=lambda x: middle[x])

    print(' '.join(map(str, order)))


if __name__ == "__main__":
    main()