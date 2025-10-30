# YOUR CODE HERE
def count_polish_sequences(N, A):
    MOD = 998244353
    
    # dp[i] will store the number of Polish sequences that can be formed with the first i elements
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: empty sequence is trivially Polish
    
    # Stack to maintain the current state of Polish sequences
    stack = []
    
    for i in range(N):
        # Current value
        v = A[i]
        
        # New stack to build the next state
        new_stack = []
        
        # Process the current stack
        while stack:
            count, total = stack.pop()
            if count < v:
                # If the current count is less than v, we can extend it
                new_stack.append((count + 1, total))
            if count == v:
                # If the current count equals v, we can complete a Polish sequence
                dp[i + 1] = (dp[i + 1] + total) % MOD
        
        # Always add the new sequence starting with v
        new_stack.append((1, dp[i]))
        
        # Update the stack
        stack = new_stack
    
    # The result is the number of Polish sequences that can be formed with all N elements
    return dp[N]

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

result = count_polish_sequences(N, A)
print(result)