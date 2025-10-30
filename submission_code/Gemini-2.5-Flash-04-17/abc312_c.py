import sys
import bisect

# Read N and M
N, M = map(int, sys.stdin.readline().split())

# Read arrays A and B
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))

# Sort A and B
A.sort()
B.sort()

# Function to check the condition for a given price X
# Condition: number of people who may sell an apple for X yen is >= number of people who may buy an apple for X yen
# A seller i may sell for X if A_i <= X.
# A buyer j may buy for X if B_j >= X.
def check(X):
    # Number of sellers willing to sell for X: count A_i <= X
    # In sorted A, this is the index returned by bisect_right
    sellers_willing = bisect.bisect_right(A, X)

    # Number of buyers willing to buy for X: count B_j >= X
    # In sorted B, this is M - count(B_j < X)
    # count(B_j < X) is the index returned by bisect_left
    buyers_willing = M - bisect.bisect_left(B, X)

    return sellers_willing >= buyers_willing

# Binary search for the minimum integer X that satisfies the condition.
# The range of possible prices for X is from 1 up to a value that guarantees the condition is met.
# The minimum possible price is 1 (since A_i, B_j >= 1).
# A price X = 10^9 + 1 is strictly greater than any A_i or B_j (since A_i, B_j <= 10^9).
# For X = 10^9 + 1, S(X) = count(A_i <= 10^9 + 1) = N (all sellers).
# For X = 10^9 + 1, B(X) = count(B_j >= 10^9 + 1) = 0 (no buyers).
# Since N >= 1 (as per constraints), S(X) = N >= 0 = B(X) is always true for X = 10^9 + 1.
# So, 10^9 + 1 is an upper bound for the search space that guarantees the condition is met.
low = 1
high = 10**9 + 1
ans = high # Initialize ans to a value that is guaranteed to satisfy the condition (like the upper bound)

while low <= high:
    mid = low + (high - low) // 2 # Calculate midpoint safely

    if check(mid):
        # mid satisfies the condition. It might be the minimum answer.
        ans = mid # Update potential answer
        high = mid - 1 # Try smaller values in the left half [low, mid-1]
    else:
        # mid does not satisfy the condition. Need a higher price.
        low = mid + 1 # Search in the right half [mid+1, high]

# Print the minimum price found
print(ans)