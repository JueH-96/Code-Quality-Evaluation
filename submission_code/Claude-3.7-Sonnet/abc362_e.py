def main():
    N = int(input())
    A = list(map(int, input().split()))
    
    MOD = 998244353
    
    # For k = 1, the answer is N (every single element is an arithmetic sequence)
    results = [N]
    
    if N == 1:
        print(results[0])
        return
    
    # For k = 2, the answer is C(N, 2) (every pair forms an arithmetic sequence)
    results.append((N * (N - 1) // 2) % MOD)
    
    if N == 2:
        print(" ".join(map(str, results)))
        return
    
    # Using a dictionary to store dp[j][d][l], which represents the number of 
    # arithmetic subsequences of length l ending at position j with common difference d.
    dp = {}
    
    # Initialization for length 2.
    for j in range(N):
        for i in range(j):
            d = A[j] - A[i]
            if j not in dp:
                dp[j] = {}
            if d not in dp[j]:
                dp[j][d] = {}
            if 2 not in dp[j][d]:
                dp[j][d][2] = 0
            dp[j][d][2] += 1
    
    # Compute for lengths 3 to N.
    for l in range(3, N+1):
        next_dp = {}
        for j in range(N):
            if j in dp:
                for d in dp[j]:
                    if l-1 in dp[j][d]:
                        for k in range(j+1, N):
                            if A[k] - A[j] == d:
                                if k not in next_dp:
                                    next_dp[k] = {}
                                if d not in next_dp[k]:
                                    next_dp[k][d] = {}
                                if l not in next_dp[k][d]:
                                    next_dp[k][d][l] = 0
                                next_dp[k][d][l] = (next_dp[k][d][l] + dp[j][d][l-1]) % MOD
        
        # Sum up the count for length l.
        count = 0
        for i in next_dp:
            for d in next_dp[i]:
                if l in next_dp[i][d]:
                    count = (count + next_dp[i][d][l]) % MOD
        
        results.append(count)
        dp = next_dp
    
    print(" ".join(map(str, results)))

if __name__ == "__main__":
    main()