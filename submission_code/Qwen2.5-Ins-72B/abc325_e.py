import sys

def min_travel_time(N, A, B, C, D):
    # Initialize the DP table
    dp = [[float('inf')] * N for _ in range(N)]
    
    # Base case: travel time from city i to itself is 0
    for i in range(N):
        dp[i][i] = 0
    
    # Fill the DP table for company car travel
    for i in range(N):
        for j in range(N):
            if i != j:
                dp[i][j] = D[i][j] * A
    
    # Floyd-Warshall algorithm to find the shortest path using company car
    for k in range(N):
        for i in range(N):
            for j in range(N):
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])
    
    # Initialize the result with the company car travel time from 1 to N
    result = dp[0][N-1]
    
    # Check if switching to train at any city can reduce the travel time
    for k in range(N):
        if k != 0 and k != N-1:
            train_time = dp[0][k] + D[k][N-1] * B + C
            result = min(result, train_time)
    
    return result

# Read input
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = int(data[1])
B = int(data[2])
C = int(data[3])
D = [list(map(int, data[4 + i * N: 4 + (i + 1) * N])) for i in range(N)]

# Solve the problem
result = min_travel_time(N, A, B, C, D)

# Print the result
print(result)