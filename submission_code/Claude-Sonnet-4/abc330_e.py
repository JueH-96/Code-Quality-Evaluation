# YOUR CODE HERE
N, Q = map(int, input().split())
A = list(map(int, input().split()))

# Count frequency of numbers in range [0, N]
# MEX will be at most N, so we only need to track up to N
freq = [0] * (N + 1)

# Initialize frequency count
for x in A:
    if 0 <= x <= N:
        freq[x] += 1

def find_mex():
    # Find the first number with frequency 0
    for i in range(N + 1):
        if freq[i] == 0:
            return i
    return N + 1  # This shouldn't happen given the constraints

# Process queries
for _ in range(Q):
    i, x = map(int, input().split())
    i -= 1  # Convert to 0-indexed
    
    # Remove the old value from frequency count
    old_val = A[i]
    if 0 <= old_val <= N:
        freq[old_val] -= 1
    
    # Add the new value to frequency count
    A[i] = x
    if 0 <= x <= N:
        freq[x] += 1
    
    # Find and print MEX
    print(find_mex())