MOD = 998244353

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_A = sys.stdin.read().split()
    N = int(N_and_A[0])
    A = list(map(int, N_and_A[1:N+1]))
    initial = [i % 2 for i in range(1, N+1)]
    
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: one way to have an empty sequence
    
    stack = []
    for r in range(1, N+1):
        # Pop elements from the stack that don't extend the current range
        while stack and (r == N or A[r-1] != A[stack[-1][0]-1]):
            _, top_r, ways = stack.pop()
            if stack:
                l, _, prev_ways = stack[-1]
                dp[r] = (dp[r] + prev_ways * ways) % MOD
            else:
                dp[r] = (dp[r] + ways) % MOD
        if not stack or A[r-1] == A[stack[-1][0]-1]:
            if stack:
                l, _, ways = stack[-1]
                if A[r-1] == A[l-1]:
                    stack.append((l, r, ways))
                else:
                    stack.append((r, r, dp[r-1]))
            else:
                stack.append((1, r, dp[0]))
        else:
            stack.append((r, r, dp[r-1]))
    
    # Finalize the DP array by considering the entire range
    while stack:
        _, top_r, ways = stack.pop()
        if stack:
            l, _, prev_ways = stack[-1]
            dp[N] = (dp[N] + prev_ways * ways) % MOD
        else:
            dp[N] = (dp[N] + ways) % MOD
    
    print(dp[N])

if __name__ == "__main__":
    main()