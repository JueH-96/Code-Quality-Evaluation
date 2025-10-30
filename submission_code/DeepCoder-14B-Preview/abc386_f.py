def can_transform(K, S, T):
    len_S = len(S)
    len_T = len(T)
    
    # If the difference in lengths is more than K, immediately return False
    if abs(len_S - len_T) > K:
        return False
    
    # Create a 2D DP table with (len_S + 1) rows and (len_T + 1) columns
    dp = [[0] * (len_T + 1) for _ in range(len_S + 1)]
    
    # Base cases
    for i in range(len_S + 1):
        dp[i][0] = i  # Deleting all characters from S to match empty T
    for j in range(len_T + 1):
        dp[0][j] = j  # Inserting all characters into empty S to match T
    
    # Fill the DP table
    for i in range(1, len_S + 1):
        for j in range(1, len_T + 1):
            # Calculate the cost of substitution
            cost = 0 if S[i-1] == T[j-1] else 1
            # Determine the minimum operations
            dp[i][j] = min(
                dp[i-1][j] + 1,     # Delete operation
                dp[i][j-1] + 1,     # Insert operation
                dp[i-1][j-1] + cost # Replace operation
            )
            # Early exit if current dp[i][j] exceeds K
            if dp[i][j] > K:
                # To optimize, we can break early if possible
                # However, in Python, breaking early in nested loops is tricky
                # We'll proceed but keep an eye on the value
                pass
    
    return dp[len_S][len_T] <= K

# Read input
K = int(input())
S = input().strip()
T = input().strip()

# Determine and print the result
if can_transform(K, S, T):
    print("Yes")
else:
    print("No")