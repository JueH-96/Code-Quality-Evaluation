# YOUR CODE HERE
def min_cost_to_construct_string(T, N, bags):
    # Length of the target string
    L = len(T)
    
    # Initialize dp array
    dp = [float('inf')] * (L + 1)
    dp[0] = 0  # Base case: cost to construct empty string is 0
    
    # Iterate over each position in T
    for i in range(L):
        if dp[i] == float('inf'):
            continue  # If this position is unreachable, skip it
        
        # Try to extend the current prefix T[0:i] using strings from bags
        for bag in bags:
            for s in bag:
                # Check if we can append s to T[0:i] to form a prefix of T
                if T.startswith(s, i):
                    # Update dp[i + len(s)]
                    dp[i + len(s)] = min(dp[i + len(s)], dp[i] + 1)
    
    # The answer is the cost to construct T[0:L]
    return dp[L] if dp[L] != float('inf') else -1

import sys
input = sys.stdin.read
data = input().strip().split('
')

T = data[0].strip()
N = int(data[1].strip())
bags = []

for i in range(2, 2 + N):
    line = data[i].strip().split()
    A_i = int(line[0])
    strings = line[1:]
    bags.append(strings)

result = min_cost_to_construct_string(T, N, bags)
print(result)