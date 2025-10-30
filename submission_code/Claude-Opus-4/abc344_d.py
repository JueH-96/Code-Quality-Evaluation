# YOUR CODE HERE
T = input().strip()
N = int(input())

bags = []
for _ in range(N):
    parts = input().split()
    A = int(parts[0])
    strings = parts[1:A+1]
    bags.append(strings)

# dp[i][j] = minimum cost to form T[0:j] using first i bags
# Initialize with infinity
INF = float('inf')
dp = [[INF] * (len(T) + 1) for _ in range(N + 1)]

# Base case: empty string with 0 bags costs 0
dp[0][0] = 0

# Fill the DP table
for i in range(1, N + 1):
    # Option 1: Skip bag i
    for j in range(len(T) + 1):
        dp[i][j] = dp[i-1][j]
    
    # Option 2: Select a string from bag i
    for string in bags[i-1]:
        for j in range(len(T) + 1):
            if dp[i-1][j] < INF:
                # Check if we can append this string
                if j + len(string) <= len(T) and T[j:j+len(string)] == string:
                    dp[i][j+len(string)] = min(dp[i][j+len(string)], dp[i-1][j] + 1)

# The answer is the minimum cost to form the entire string T
answer = dp[N][len(T)]
if answer == INF:
    print(-1)
else:
    print(answer)