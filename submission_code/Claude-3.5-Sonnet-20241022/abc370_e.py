def solve():
    MOD = 998244353
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    # dp[i][j] = number of valid divisions for prefix A[0:i] where j is whether we can use this position as split point
    dp = [[0] * 2 for _ in range(N+1)]
    dp[0][1] = 1
    
    # For each position, compute cumulative sums starting from that position
    sums = []
    for i in range(N):
        curr_sum = 0
        curr_sums = []
        for j in range(i, N):
            curr_sum += A[j]
            curr_sums.append(curr_sum)
        sums.append(curr_sums)
    
    # For each position
    for i in range(N):
        # For each previous state (can split or not)
        for prev in range(2):
            if dp[i][prev] == 0:
                continue
                
            # Try not splitting at current position
            can_extend = 1
            # Check if extending creates a subsequence with sum K
            if prev == 1:  # If we could split at previous position
                if sums[i][0] == K:  # Check single element
                    can_extend = 0
            dp[i+1][0] = (dp[i+1][0] + dp[i][prev] * can_extend) % MOD
            
            # Try splitting at current position
            can_split = 1
            # Check all possible subsequences ending at current position
            for j in range(i+1):
                if prev == 0 and j < i:
                    continue
                if sums[j][i-j] == K:
                    can_split = 0
                    break
            dp[i+1][1] = (dp[i+1][1] + dp[i][prev] * can_split) % MOD
    
    # Final answer is sum of both possibilities at position N
    ans = sum(dp[N]) % MOD
    print(ans)

solve()