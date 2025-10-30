def min_cost_to_form_string(T, N, bags):
    from collections import defaultdict, deque

    # Initialize a DP array where dp[i] is the minimum cost to form T[:i]
    dp = [float('inf')] * (len(T) + 1)
    dp[0] = 0  # Cost to form an empty string is 0

    # Process each bag
    for i in range(N):
        current_bag = bags[i]
        # We need to iterate backwards to avoid overwriting results we still need to use
        for j in range(len(T), -1, -1):
            for string in current_bag:
                # If we can concatenate this string to form T[:j+len(string)]
                if j + len(string) <= len(T) and T[j:j + len(string)] == string:
                    dp[j + len(string)] = min(dp[j + len(string)], dp[j] + 1)

    # The answer is the minimum cost to form the entire string T
    return dp[len(T)] if dp[len(T)] != float('inf') else -1


# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

T = data[0]
N = int(data[1])
bags = []

for i in range(2, 2 + N):
    parts = data[i].split()
    A_i = int(parts[0])
    strings = parts[1:A_i + 1]
    bags.append(strings)

# Get the result
result = min_cost_to_form_string(T, N, bags)

# Print the result
print(result)