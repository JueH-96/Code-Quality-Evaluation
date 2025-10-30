# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

total = 0

# For each starting position l
for l in range(N):
    current_sum = 0
    # For each ending position r >= l
    for r in range(l, N):
        # Add the next element to extend the subarray
        current_sum += A[r]
        # Add the modulo of this subarray sum to our total
        total += current_sum % M

print(total)