N, K = map(int, input().split())
P = list(map(int, input().split()))

# Create a mapping from value to index (0-based)
pos = {}
for i in range(N):
    pos[P[i]] = i

min_span = float('inf')

# Try each possible starting value a for consecutive integers a, a+1, ..., a+K-1
for a in range(1, N - K + 2):
    positions = []
    for val in range(a, a + K):
        positions.append(pos[val])
    
    # Calculate span as max position - min position
    span = max(positions) - min(positions)
    min_span = min(min_span, span)

print(min_span)