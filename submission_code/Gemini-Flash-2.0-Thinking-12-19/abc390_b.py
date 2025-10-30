import sys

# Read N
N = int(sys.stdin.readline())

# Read the sequence A
a_list = list(map(int, sys.stdin.readline().split()))

# A sequence with N=2 is always a geometric progression by definition.
# The loop below handles N=2 correctly because range(2, 2) is empty.
# For N > 2, we need to check if the ratio of consecutive terms is constant.
# A sequence A_1, A_2, ..., A_N is a geometric progression if
# A_k = A_{k-1} * r for some constant ratio r, for all k from 2 to N.
# This implies A_k / A_{k-1} = r for all k from 2 to N.
# So, A_2/A_1 = A_3/A_2 = ... = A_N/A_{N-1}.
# We can check this by comparing the ratio of each consecutive pair (A_k, A_{k-1})
# with the ratio of the first pair (A_2, A_1).
# Check if A_k / A_{k-1} = A_2 / A_1 for k from 3 to N.
# Using cross-multiplication to avoid floating-point issues:
# A_k * A_1 = A_{k-1} * A_2 for k from 3 to N.
# Using 0-based indexing for the list `a_list` (where a_list[i] is A_{i+1}):
# A_1 is a_list[0]
# A_2 is a_list[1]
# A_{k-1} is a_list[k-2]
# A_k is a_list[k-1]
# The condition A_k * A_1 = A_{k-1} * A_2 translates to
# a_list[k-1] * a_list[0] == a_list[k-2] * a_list[1] for k from 3 to N.
# Let the loop index be i, corresponding to the index of A_k in 0-based list.
# A_k is a_list[i], so i = k-1.
# As k goes from 3 to N, i goes from 2 to N-1.
# The condition becomes a_list[i] * a_list[0] == a_list[i-1] * a_list[1] for i from 2 to N-1.

is_gp = True
# Loop starts from the third element (index 2 in 0-based list)
# We compare the ratio A[i]/A[i-1] with the initial ratio A[1]/A[0]
for i in range(2, N):
    # Check if A[i] / A[i-1] == A[1] / A[0]
    # Using cross-multiplication: a_list[i] * a_list[0] == a_list[i-1] * a_list[1]
    # All elements a_list[j] are guaranteed to be >= 1 (constraint 1 <= A_i <= 10^9),
    # so a_list[0] and a_list[i-1] are not zero.
    # Python handles large integers automatically, so products up to 10^18 are fine.
    if a_list[i] * a_list[0] != a_list[i-1] * a_list[1]:
        is_gp = False
        break

# Print the result
if is_gp:
    print("Yes")
else:
    print("No")