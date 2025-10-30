# YOUR CODE HERE
N = int(input())

# Read the edge weights
D = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N):
    row = list(map(int, input().split()))
    for j in range(len(row)):
        D[i][i + j + 1] = row[j]

# dp[mask] = maximum weight when vertices in mask are used
dp = [0] * (1 << N)

# Try all possible states
for mask in range(1 << N):
    # Find the first two unused vertices
    unused = []
    for i in range(N):
        if not (mask & (1 << i)):
            unused.append(i)
    
    if len(unused) < 2:
        continue
    
    # Try pairing the first unused vertex with all other unused vertices
    first = unused[0]
    for j in range(1, len(unused)):
        second = unused[j]
        new_mask = mask | (1 << first) | (1 << second)
        
        # Convert to 1-indexed for the weight lookup
        v1, v2 = first + 1, second + 1
        if v1 > v2:
            v1, v2 = v2, v1
        
        weight = D[v1][v2]
        dp[new_mask] = max(dp[new_mask], dp[mask] + weight)

print(max(dp))