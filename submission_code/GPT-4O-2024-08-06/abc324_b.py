# YOUR CODE HERE
def can_be_expressed_as_2x3y(N):
    # Factor out powers of 2
    while N % 2 == 0:
        N //= 2
    
    # Factor out powers of 3
    while N % 3 == 0:
        N //= 3
    
    # If N is reduced to 1, it means it was composed only of 2's and 3's
    if N == 1:
        return "Yes"
    else:
        return "No"

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Output the result
print(can_be_expressed_as_2x3y(N))