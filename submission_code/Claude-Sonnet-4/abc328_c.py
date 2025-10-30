# YOUR CODE HERE
N, Q = map(int, input().split())
S = input().strip()

# Preprocess: create array where consecutive[i] = 1 if S[i] == S[i+1], 0 otherwise
# Using 0-based indexing internally
consecutive = [0] * N
for i in range(N-1):
    if S[i] == S[i+1]:
        consecutive[i] = 1

# Create prefix sum array for quick range queries
prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i+1] = prefix_sum[i] + consecutive[i]

# Process queries
for _ in range(Q):
    l, r = map(int, input().split())
    # Convert to 0-based indexing
    l -= 1
    r -= 1
    
    # We need to count consecutive pairs in range [l, r-1]
    # This corresponds to prefix_sum[r] - prefix_sum[l]
    if l >= r:
        print(0)
    else:
        result = prefix_sum[r] - prefix_sum[l]
        print(result)