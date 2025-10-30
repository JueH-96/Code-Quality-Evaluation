def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = data[2]
    T = data[3]

    # dp[i] will be True if we can cover (transform from '#') the substring S[0:i] 
    # using blocks that match T.
    dp = [False] * (N + 1)
    dp[0] = True

    for i in range(N):
        if dp[i] and i + M <= N:
            # Check if S[i:i+M] matches T
            if S[i:i+M] == T:
                dp[i + M] = True

    print("Yes" if dp[N] else "No")

def main():
    solve()

if __name__ == "__main__":
    main()