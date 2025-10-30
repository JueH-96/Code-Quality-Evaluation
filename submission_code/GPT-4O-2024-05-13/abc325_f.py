# YOUR CODE HERE
def min_cost_to_monitor_sections(N, D, L1, C1, K1, L2, C2, K2):
    # Initialize a large number for infinity
    INF = float('inf')
    
    # Initialize dp array where dp[i] represents the minimum cost to monitor a section of length i
    dp = [INF] * (max(D) + 1)
    dp[0] = 0  # No cost to monitor a section of length 0
    
    # Update dp array for type-1 sensors
    for i in range(1, max(D) + 1):
        for k in range(1, K1 + 1):
            if i - k * L1 >= 0:
                dp[i] = min(dp[i], dp[i - k * L1] + k * C1)
            else:
                break
    
    # Update dp array for type-2 sensors
    for i in range(1, max(D) + 1):
        for k in range(1, K2 + 1):
            if i - k * L2 >= 0:
                dp[i] = min(dp[i], dp[i - k * L2] + k * C2)
            else:
                break
    
    # Check if it's possible to monitor all sections and calculate the total cost
    total_cost = 0
    for length in D:
        if dp[length] == INF:
            return -1
        total_cost += dp[length]
    
    return total_cost

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = list(map(int, data[1:N+1]))
L1, C1, K1 = map(int, data[N+1:N+4])
L2, C2, K2 = map(int, data[N+4:N+7])

# Calculate and print the result
result = min_cost_to_monitor_sections(N, D, L1, C1, K1, L2, C2, K2)
print(result)