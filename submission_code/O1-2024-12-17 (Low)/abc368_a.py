def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    # Take the bottom K cards and put them on top
    B = A[-K:] + A[:-K]

    print(' '.join(map(str, B)))

# Do not remove the call to main()
main()