# YOUR CODE HERE
def min_cost_to_form_string(T, N, bags):
    # Initialize dp array with a large number (infinity)
    dp = [float('inf')] * (len(T) + 1)
    dp[0] = 0  # Base case: no cost to form an empty string

    for i in range(N):
        for j in range(len(T), -1, -1):
            if dp[j] != float('inf'):
                for s in bags[i]:
                    if T[j:j+len(s)] == s:
                        dp[j+len(s)] = min(dp[j+len(s)], dp[j] + 1)

    return dp[len(T)] if dp[len(T)] != float('inf') else -1

# Read input
import sys
input = sys.stdin.read
data = input().split()

T = data[0]
N = int(data[1])
index = 2
bags = []

for _ in range(N):
    A_i = int(data[index])
    bag = data[index+1:index+1+A_i]
    bags.append(bag)
    index += 1 + A_i

# Calculate the minimum cost
result = min_cost_to_form_string(T, N, bags)
print(result)