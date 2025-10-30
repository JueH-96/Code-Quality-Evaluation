n, k = map(int, input().split())
p = list(map(int, input().split()))

# Create a mapping from value to its position (1-indexed)
pos = {}
for i in range(n):
    pos[p[i]] = i + 1

min_span = float('inf')

# Try all possible consecutive sequences of length k
for a in range(1, n - k + 2):  # a can be from 1 to n-k+1
    # Get positions of consecutive integers a, a+1, ..., a+k-1
    positions = []
    for val in range(a, a + k):
        positions.append(pos[val])
    
    # Sort positions to get them in increasing order
    positions.sort()
    
    # Calculate span
    span = positions[-1] - positions[0]
    min_span = min(min_span, span)

print(min_span)