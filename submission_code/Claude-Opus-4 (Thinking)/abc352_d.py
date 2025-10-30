# YOUR CODE HERE
N, K = map(int, input().split())
P = list(map(int, input().split()))

# Create a mapping from value to index (1-indexed)
value_to_index = {}
for i in range(N):
    value_to_index[P[i]] = i + 1

min_diff = float('inf')

# Try all possible sets of K consecutive integers
for a in range(1, N - K + 2):  # a can go from 1 to N-K+1
    # The set is {a, a+1, ..., a+K-1}
    min_index = float('inf')
    max_index = -float('inf')
    
    for val in range(a, a + K):
        idx = value_to_index[val]
        min_index = min(min_index, idx)
        max_index = max(max_index, idx)
    
    # Calculate i_K - i_1
    diff = max_index - min_index
    min_diff = min(min_diff, diff)

print(min_diff)