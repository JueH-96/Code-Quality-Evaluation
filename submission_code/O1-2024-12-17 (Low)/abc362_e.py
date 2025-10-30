def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353
    
    # Edge case: if N == 1, only one subsequence of length 1
    if N == 1:
        print(1)
        return
    
    # ans[k] will hold the count of arithmetic subsequences of length k (mod MOD)
    ans = [0] * (N+1)
    # All single elements form length-1 subsequences (k=1), so there are N of them
    ans[1] = N
    
    # dp[i] will be a dictionary: diff -> [counts for lengths 0..N]
    # dp[i][diff][k] = number of arithmetic subsequences ending at index i
    # with common difference diff and length k
    from collections import defaultdict
    
    dp = [defaultdict(lambda: [0]*(N+1)) for _ in range(N)]
    
    # Build DP
    for i in range(N):
        for j in range(i):
            diff = A[i] - A[j]
            # Extend all subsequences that end at j with difference diff
            # to one more element (which is i)
            for k in range(1, N+1):
                count_at_j = dp[j][diff][k]
                if count_at_j:
                    dp[i][diff][k+1] = (dp[i][diff][k+1] + count_at_j) % MOD
            
            # Any pair (j, i) forms a subsequence of length 2 with difference diff
            dp[i][diff][2] = (dp[i][diff][2] + 1) % MOD
    
    # Sum up counts from dp for k >= 2
    for i in range(N):
        for diff, arr in dp[i].items():
            for k in range(2, N+1):
                ans[k] = (ans[k] + arr[k]) % MOD

    # Print the results for k=1..N
    print(" ".join(str(ans[k] % MOD) for k in range(1, N+1)))

# Do not forget to call main at the end
if __name__ == "__main__":
    main()