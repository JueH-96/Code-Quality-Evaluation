from collections import defaultdict

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Initialize dp: a list of defaultdicts, each mapping difference d to defaultdict(int) for lengths
    dp = [defaultdict(lambda: defaultdict(int)) for _ in range(N)]
    
    for i in range(N):
        for j in range(i):
            d = A[i] - A[j]
            # Update dp[i] based on dp[j]
            for l in dp[j][d]:
                cnt = dp[j][d][l]
                dp[i][d][l+1] += cnt
            # Add new subsequence of length 2 (j, i)
            dp[i][d][2] += 1
    
    # Compute the answers
    ans = [0] * (N + 1)
    ans[1] = N  # All elements are valid for k=1
    for k in range(2, N+1):
        total = 0
        for i in range(N):
            for d in dp[i]:
                total += dp[i][d][k]
        ans[k] = total % MOD
    
    print(' '.join(map(str, ans[1:N+1])))

if __name__ == "__main__":
    main()