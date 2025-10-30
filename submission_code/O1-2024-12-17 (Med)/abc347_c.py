def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, A, B = map(int, data[:3])
    D = list(map(int, data[3:]))

    K = A + B  # total days in one "week"
    # Compute each D_i modulo K
    S = [d % K for d in D]
    # Sort these modulo values
    S.sort()

    # Find the largest gap between consecutive elements in circular order
    G = 0
    for i in range(N - 1):
        G = max(G, S[i + 1] - S[i])
    # Also consider the gap that wraps around from S[-1] back to S[0]
    G = max(G, (S[0] + K) - S[-1])

    # The minimal inclusive arc covering all points is (K - G + 1)
    # If that arc size is within the holiday length A, print Yes, else No
    if (K - G + 1) <= A:
        print("Yes")
    else:
        print("No")

# Do not forget to call main()
main()