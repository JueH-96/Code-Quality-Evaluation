import sys
import bisect

# Read N, M, D
line1 = sys.stdin.readline().split()
N = int(line1[0])
M = int(line1[1])
D = int(line1[2])

# Read array A
A = list(map(int, sys.stdin.readline().split()))

# Read array B
B = list(map(int, sys.stdin.readline().split()))

# Sort A and B
A.sort()
B.sort()

# Initialize max_sum
max_sum = -1

# Iterate through each element in A
for a in A:
    # We need to find b in B such that a - D <= b <= a + D
    # Find the insertion point for a + D in B using bisect_right.
    # bisect_right(B, value) returns an index i such that all elements
    # in B[:i] are <= value, and all elements in B[i:] are > value.
    # So, B[i-1] (if i > 0) is the largest element in B <= value.
    
    upper_bound_for_b = a + D
    upper_bound_idx = bisect.bisect_right(B, upper_bound_for_b)

    # The largest element in B less than or equal to a + D is at index upper_bound_idx - 1.
    # We check if such an index is valid (i.e., upper_bound_idx > 0).
    if upper_bound_idx > 0:
        candidate_b = B[upper_bound_idx - 1]

        # Check if this candidate_b also satisfies the lower bound condition: b >= a - D
        lower_bound_for_b = a - D
        if candidate_b >= lower_bound_for_b:
            # We found a valid pair (a, candidate_b). Update max_sum.
            # Use max() to find the overall maximum sum.
            max_sum = max(max_sum, a + candidate_b)

# Print the result
print(max_sum)