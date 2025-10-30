import sys

# Read input
data = sys.stdin.read().splitlines()
K = int(data[0])
S = data[1]
T = data[2]
len_S = len(S)
len_T = len(T)

# Check length difference
if abs(len_S - len_T) > K:
    print("No")
else:
    INF = 10**9  # A large number to represent infinity
    # Initialize prev_dp for i=0 (distance from empty string to T[:j])
    prev_dp = [j for j in range(len_T + 1)]
    
    # Iterate through each character in S
    for i in range(1, len_S + 1):
        # Create a new dp row initialized to INF
        curr_dp = [INF] * (len_T + 1)
        # Cost to delete i characters to match empty string
        curr_dp[0] = i
        
        # Define the band limits
        low = max(0, i - K)
        high = min(len_T, i + K)
        start_j = max(1, low)  # Start j from max(1, low) since j=0 is handled
        
        # Compute dp values within the band
        for j in range(start_j, high + 1):
            if S[i-1] == T[j-1]:
                cost = 0
            else:
                cost = 1
            
            # dp[i-1][j]
            if abs(i - 1 - j) > K:
                dp_im1_j = INF
            else:
                dp_im1_j = prev_dp[j]
            delete_cost = dp_im1_j + 1
            
            # dp[i][j-1]
            if low <= j - 1 <= high:
                dp_i_jm1 = curr_dp[j - 1]
            else:
                dp_i_jm1 = INF
            insert_cost = dp_i_jm1 + 1
            
            # dp[i-1][j-1]
            if abs(i - j) > K:  # abs((i-1) - (j-1)) = abs(i - j)
                dp_im1_jm1 = INF
            else:
                dp_im1_jm1 = prev_dp[j - 1]
            if cost == 0:
                replace_cost = dp_im1_jm1
            else:
                replace_cost = dp_im1_jm1 + 1
            
            # Minimum of delete, insert, replace costs
            curr_dp[j] = min(delete_cost, insert_cost, replace_cost)
        
        # Set prev_dp to current dp for next iteration
        prev_dp = curr_dp
    
    # The distance is at dp[len_S][len_T]
    distance = prev_dp[len_T]
    
    # Check if distance <= K
    if distance <= K:
        print("Yes")
    else:
        print("No")