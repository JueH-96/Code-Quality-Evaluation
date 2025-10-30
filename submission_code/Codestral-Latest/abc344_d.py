import sys
from collections import defaultdict

def min_cost_to_form_string(T, N, bags):
    # Initialize a dictionary to store the minimum cost to form each prefix of T
    dp = defaultdict(lambda: float('inf'))
    dp[0] = 0  # Base case: cost to form an empty string is 0

    for i in range(1, len(T) + 1):
        for j in range(1, N + 1):
            for string in bags[j]:
                if T[:i].endswith(string):
                    dp[i] = min(dp[i], dp[i - len(string)] + 1)

    return dp[len(T)] if dp[len(T)] != float('inf') else -1

# Read input
T = input().strip()
N = int(input().strip())
bags = {}
for i in range(1, N + 1):
    line = input().strip().split()
    A_i = int(line[0])
    bags[i] = line[1:]

# Calculate and print the result
result = min_cost_to_form_string(T, N, bags)
print(result)