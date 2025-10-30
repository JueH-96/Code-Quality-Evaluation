def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Take the last K cards and move them to the front
    B = A[N-K:] + A[:N-K]

    # Print the resulting stack from top to bottom
    print(" ".join(map(str, B)))

if __name__ == "__main__":
    main()