# YOUR CODE HERE
S = input().strip()
n = len(S)

# We need to find minimum insertions to make the string follow pattern "ioioio..."
# The final length must be even and at least n

min_insertions = float('inf')

# Try all possible even lengths from n to 2*n (worst case we need to insert n characters)
for target_len in range(n, 2*n + 1):
    if target_len % 2 == 1:
        continue
    
    # Generate target pattern of length target_len
    target = ""
    for i in range(target_len):
        if i % 2 == 0:
            target += 'i'
        else:
            target += 'o'
    
    # dp[i][j] = minimum insertions to match S[0:i] with target[0:j]
    dp = [[float('inf')] * (target_len + 1) for _ in range(n + 1)]
    
    # Base case: empty string matches empty target with 0 insertions
    dp[0][0] = 0
    
    # If we've used 0 characters from S, we need j insertions to match j characters of target
    for j in range(1, target_len + 1):
        dp[0][j] = j
    
    # Fill the dp table
    for i in range(1, n + 1):
        for j in range(1, target_len + 1):
            # Option 1: Insert target[j-1] without using S[i-1]
            dp[i][j] = min(dp[i][j], dp[i][j-1] + 1)
            
            # Option 2: If S[i-1] matches target[j-1], use it
            if S[i-1] == target[j-1]:
                dp[i][j] = min(dp[i][j], dp[i-1][j-1])
    
    # The answer for this target length is dp[n][target_len]
    min_insertions = min(min_insertions, dp[n][target_len])

print(min_insertions)