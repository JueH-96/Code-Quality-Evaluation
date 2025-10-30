# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    strings = data[1:]
    
    results = []
    
    for k in range(N):
        T = strings[k]
        len_T = len(T)
        
        # Cost to make T empty
        min_cost = len_T
        
        # Compare T with each of the previous strings
        for j in range(k):
            S = strings[j]
            len_S = len(S)
            
            # DP table for edit distance
            dp = [[0] * (len_S + 1) for _ in range(len_T + 1)]
            
            # Initialize base cases
            for i in range(len_T + 1):
                dp[i][0] = i  # Cost to make T[:i] empty
            for j in range(len_S + 1):
                dp[0][j] = j  # Cost to make empty T into S[:j]
            
            # Fill the DP table
            for i in range(1, len_T + 1):
                for j in range(1, len_S + 1):
                    if T[i-1] == S[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(dp[i-1][j] + 1,  # Delete from T
                                       dp[i][j-1] + 1)  # Add to T
            
            # Minimum cost to transform T into S
            min_cost = min(min_cost, dp[len_T][len_S])
        
        results.append(min_cost)
    
    for result in results:
        print(result)