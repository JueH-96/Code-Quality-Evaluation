# YOUR CODE HERE
n, k = map(int, input().split())
p = list(map(int, input().split()))

# Create a position map: value -> index (1-indexed)
pos = {}
for i in range(n):
    pos[p[i]] = i + 1

min_span = float('inf')

# Try all possible consecutive sequences of length k
for start in range(1, n - k + 2):
    # Check if we can form a sequence [start, start+1, ..., start+k-1]
    indices = []
    valid = True
    
    for val in range(start, start + k):
        if val in pos:
            indices.append(pos[val])
        else:
            valid = False
            break
    
    if valid:
        # Calculate the span (max_index - min_index)
        span = max(indices) - min(indices)
        min_span = min(min_span, span)

print(min_span)