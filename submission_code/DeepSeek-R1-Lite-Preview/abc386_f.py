def can_convert(S, T, K):
    len_S = len(S)
    len_T = len(T)
    
    # Early check for length difference
    if abs(len_S - len_T) > K:
        return "No"
    
    # Initialize DP arrays
    prev_dp = [i for i in range(len_S + 1)]
    
    for j in range(1, len_T + 1):
        curr_dp = [0] * (len_S + 1)
        curr_dp[0] = j
        for i in range(1, len_S + 1):
            if S[i-1] == T[j-1]:
                curr_dp[i] = prev_dp[i-1]
            else:
                curr_dp[i] = 1 + min(prev_dp[i-1],  # replace
                                     prev_dp[i],    # delete
                                     curr_dp[i-1])  # insert
            if curr_dp[i] > K:
                curr_dp[i] = float('inf')  # Use infinity to skip unnecessary values
        prev_dp = curr_dp
    
    if prev_dp[len_S] <= K:
        return "Yes"
    else:
        return "No"

# Read input
K = int(input())
S = input().strip()
T = input().strip()

# Get and print the result
result = can_convert(S, T, K)
print(result)