# YOUR CODE HERE
import bisect

N, M, P = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort B for binary search
B.sort()

# Precompute prefix sums of B for efficiency
B_prefix = [0]
for b in B:
    B_prefix.append(B_prefix[-1] + b)

total = 0

for a in A:
    # Find how many B[j] satisfy a + B[j] <= P
    # This means B[j] <= P - a
    threshold = P - a
    
    # Binary search for the rightmost position where B[j] <= threshold
    pos = bisect.bisect_right(B, threshold)
    
    # For indices 0 to pos-1, we use actual sum a + B[j]
    # For indices pos to M-1, we use P
    
    # Sum for indices where a + B[j] <= P
    if pos > 0:
        total += a * pos + B_prefix[pos]
    
    # Sum for indices where a + B[j] > P
    if pos < M:
        total += P * (M - pos)

print(total)