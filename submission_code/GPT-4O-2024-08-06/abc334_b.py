# YOUR CODE HERE
def count_christmas_trees(A, M, L, R):
    # Calculate the first k such that A + k * M >= L
    if L > A:
        k_start = (L - A + M - 1) // M
    else:
        k_start = 0
    
    # Calculate the last k such that A + k * M <= R
    if R >= A:
        k_end = (R - A) // M
    else:
        k_end = -1
    
    # Calculate the number of valid k's
    if k_start > k_end:
        return 0
    else:
        return k_end - k_start + 1

# Read input
import sys
input = sys.stdin.read
data = input().strip().split()
A = int(data[0])
M = int(data[1])
L = int(data[2])
R = int(data[3])

# Output the result
print(count_christmas_trees(A, M, L, R))