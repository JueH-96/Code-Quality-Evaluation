def main():
    import sys

    # Read input
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N, L, R = map(int, data)

    # Initialize the sequence 1..N
    seq = list(range(1, N + 1))

    # Reverse the sub-sequence from L to R (1-indexed → 0-indexed slice)
    seq[L - 1:R] = reversed(seq[L - 1:R])

    # Output
    print(' '.join(map(str, seq)))


if __name__ == "__main__":
    main()