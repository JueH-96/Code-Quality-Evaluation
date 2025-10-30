# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort arrays for binary search optimization
A.sort()
B.sort()

# Generate all candidate values where the condition might change
candidates = set()
for a in A:
    candidates.add(a)
for b in B:
    candidates.add(b + 1)

# Sort candidates
candidates = sorted(candidates)

# Use binary search to efficiently count
import bisect

# Find minimum X that satisfies the condition
for X in candidates:
    # Count sellers with A_i <= X (willing to sell at price X)
    sellers = bisect.bisect_right(A, X)
    # Count buyers with B_i >= X (willing to buy at price X)
    buyers = M - bisect.bisect_left(B, X)
    
    if sellers >= buyers:
        print(X)
        break