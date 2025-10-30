import sys

# Helper function to calculate v2(x) for x > 0
def v2(x):
    if x == 0:
        # This case should not happen based on constraints (A_i are distinct, i >= 1).
        # A_i are strictly increasing, so A_j - A_k is non-zero for j != k.
        # A_j > 0 for j > 0, so v2(A_j) for j > 0 is well-defined.
        # A_N + 1 - A_k >= A_N + 1 - A_N = 1, so v2(A_N + 1 - A_k) is well-defined.
        # v2(0) is typically handled as infinity, but it's not needed here.
        raise ValueError("v2 is not defined for 0 in this context")
    
    # (x & -x) isolates the least significant bit (LSB).
    # The value of LSB is 2**v2(x).
    # bit_length() counts the number of bits required to represent a positive integer.
    # The bit_length() of 2**p is p + 1.
    # So bit_length() of (x & -x) is v2(x) + 1.
    # This works for positive integers.
    return (x & -x).bit_length() - 1

# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

# Candidate 1: Consider i = 2^p for a large even p.
# For F=100 folds, the k-th crease from the left is a Mountain fold if v2(k) is even.
# We want to maximize f(i) = sum_{k=1 to N} [v2(i + A_k)%2 == 0].
# Let's try i = 2^p for a large even integer p, e.g., p = 60 or more.
# A_k <= 10^18 < 2^60. Let p = 60 (or any even p >= 60 up to 98).
# The range for i is 1 <= i <= 2^100 - A_N - 1.
# For p=60, i=2^60. Since A_N <= 10^18, 2^60 > A_N.
# 2^60 <= 2^100 - A_N - 1 requires A_N + 1 <= 2^100 - 2^60. This is true as A_N is small compared to 2^100.
# So i = 2^p for even p >= 60 is a valid candidate value for i.
# f(2^p) = sum_{k=1 to N} [v2(2^p + A_k)%2 == 0].
# For k=1, A_1 = 0. Term is [v2(2^p + 0)%2 == 0] = [v2(2^p)%2 == 0]. Since p is even, this is 1.
# For k >= 2, A_k > 0 and v2(A_k) < 60 <= p.
# If v2(A_k) < p, then v2(2^p + A_k) = v2(A_k). (If v2(X) < v2(Y), then v2(X+Y)=v2(X))
# So for k >= 2, the term is [v2(A_k)%2 == 0].
candidate1 = 1 # Contribution from A_1=0 (v2(i) is even)
for k in range(1, N): # A_k for k=2 to N are A[1] to A[N-1] in 0-indexed list
    if v2(A[k]) % 2 == 0:
        candidate1 += 1

# Candidate 2: Consider i = 2^100 - A_N - 1. This is the maximum possible value for i.
# f(2^100 - A_N - 1) = sum_{k=1 to N} [v2(2^100 - A_N - 1 + A_k)%2 == 0].
# Let i_val = 2**100 - A[N-1] - 1. We need to evaluate v2(i_val + A[k]).
# i_val + A[k] = 2**100 - A[N-1] - 1 + A[k] = 2**100 + (A[k] - A[N-1] - 1).
# Let Y_k = A[k] - A[N-1] - 1.
# Since A[0]=0 <= A[k] <= A[N-1], Y_k ranges from A[0] - A[N-1] - 1 = -A[N-1] - 1 to A[N-1] - A[N-1] - 1 = -1.
# So Y_k is a negative integer, -A[N-1] - 1 <= Y_k <= -1.
# |Y_k| = A[N-1] + 1 - A[k].
# Since A[k] <= A[N-1], |Y_k| >= A[N-1] + 1 - A[N-1] = 1. So Y_k is never 0.
# A[N-1] <= 10^18, so |Y_k| <= 10^18 + 1 < 2^60. Thus v2(|Y_k|) < 60.
# We need to evaluate v2(2^100 + Y_k). Since Y_k is negative, this is |2^100 + Y_k|.
# 2^100 + Y_k = 2^100 - |Y_k|.
# Since |Y_k| < 2^60, v2(|Y_k|) < 60 < 100.
# If v2(Y_k) < 100, v2(2^100 + Y_k) = v2(Y_k). This applies here.
# v2(Y_k) = v2(-|Y_k|) = v2(|Y_k|).
# So v2(i_val + A[k]) = v2(|Y_k|) = v2(A[N-1] + 1 - A[k]).
candidate2 = 0
A_N_val = A[N-1]
for k in range(N): # A_k for k=1 to N are A[0] to A[N-1]
    diff = A_N_val + 1 - A[k] # This is |A[k] - A_N - 1| as shown above
    # diff is always >= 1 since A[k] <= A[N-1]
    if v2(diff) % 2 == 0:
        candidate2 += 1

# Compare candidates and print max
print(max(candidate1, candidate2))