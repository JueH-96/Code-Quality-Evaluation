def main():
    import sys
    input = sys.stdin.readline

    # Read inputs
    N, Q = map(int, input().split())
    S = input().rstrip()

    # Build a prefix sum array where
    # prefix[i] = number of positions j <= i such that S[j] == S[j-1].
    # We use 1-based indexing for prefix; prefix[1] = 0 always.
    prefix = [0] * (N + 1)
    for i in range(2, N + 1):
        prefix[i] = prefix[i - 1] + (1 if S[i - 1] == S[i - 2] else 0)

    # Answer queries
    out = []
    for _ in range(Q):
        l, r = map(int, input().split())
        # Number of equal-adjacent pairs in S[l..r] is
        # prefix[r] - prefix[l]
        out.append(str(prefix[r] - prefix[l]))

    # Print results
    sys.stdout.write("
".join(out))


if __name__ == "__main__":
    main()