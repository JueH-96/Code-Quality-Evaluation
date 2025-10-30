import sys

def solve_fireworks(N, M, A):
    # Initialize the result list with the maximum possible value
    result = [N] * N
    
    # Set the days when fireworks are launched to 0
    for day in A:
        result[day - 1] = 0
    
    # Fill the result list from left to right
    for i in range(1, N):
        if result[i] > 0:
            result[i] = result[i - 1] + 1
    
    # Fill the result list from right to left
    for i in range(N - 2, -1, -1):
        if result[i] > result[i + 1] + 1:
            result[i] = result[i + 1] + 1
    
    return result

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Solve the problem
result = solve_fireworks(N, M, A)

# Print the result
for days in result:
    print(days)