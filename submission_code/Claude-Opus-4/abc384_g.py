# YOUR CODE HERE
import bisect

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
K = int(input())

# Precompute prefix sums for B
B_prefix = [0]
for i in range(N):
    B_prefix.append(B_prefix[-1] + B[i])

# For each query
for _ in range(K):
    X, Y = map(int, input().split())
    
    total = 0
    
    # Create sorted version of first Y elements of B for binary search
    B_sorted = sorted(B[:Y])
    B_sorted_prefix = [0]
    for val in B_sorted:
        B_sorted_prefix.append(B_sorted_prefix[-1] + val)
    
    for i in range(X):
        # For A[i], find sum of |A[i] - B[j]| for j in [0, Y)
        # Find position where B_sorted[pos] >= A[i]
        pos = bisect.bisect_left(B_sorted, A[i])
        
        # For j < pos: A[i] > B_sorted[j], so |A[i] - B_sorted[j]| = A[i] - B_sorted[j]
        # Sum = pos * A[i] - sum(B_sorted[0:pos])
        if pos > 0:
            total += pos * A[i] - B_sorted_prefix[pos]
        
        # For j >= pos: A[i] <= B_sorted[j], so |A[i] - B_sorted[j]| = B_sorted[j] - A[i]
        # Sum = sum(B_sorted[pos:Y]) - (Y - pos) * A[i]
        if pos < Y:
            total += (B_sorted_prefix[Y] - B_sorted_prefix[pos]) - (Y - pos) * A[i]
    
    print(total)