# YOUR CODE HERE
n, m = map(int, input().split())

# Initialize adjacency matrix
# beats[i][j] = True if person i+1 beats person j+1
beats = [[False] * n for _ in range(n)]

# Read the superiority information
for _ in range(m):
    a, b = map(int, input().split())
    beats[a-1][b-1] = True

# Apply transitive closure using Floyd-Warshall
for k in range(n):
    for i in range(n):
        for j in range(n):
            if beats[i][k] and beats[k][j]:
                beats[i][j] = True

# Count how many people each person beats
strongest_candidates = []
for i in range(n):
    beat_count = sum(beats[i])
    if beat_count == n - 1:  # beats everyone else
        strongest_candidates.append(i + 1)

# Check if we can uniquely determine the strongest
if len(strongest_candidates) == 1:
    print(strongest_candidates[0])
else:
    print(-1)