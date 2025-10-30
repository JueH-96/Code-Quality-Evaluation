import sys
from bisect import bisect_left

# Fenwick Tree (Binary Indexed Tree)
class FenwickTree:
    def __init__(self, size):
        # size is the maximum value of rank (1-based)
        self._size = size
        self._tree = [0] * (size + 1)

    def update(self, idx, delta):
        # Add delta to the element at index idx (1-based)
        # idx must be between 1 and self._size
        while idx <= self._size:
            self._tree[idx] += delta
            idx += idx & (-idx)

    def query(self, idx):
        # Get the prefix sum up to index idx (1-based)
        # idx must be between 0 and self._size
        total = 0
        while idx > 0:
            total += self._tree[idx]
            idx -= idx & (-idx)
        return total

# Read input
N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

M = 10**8

# The expression is sum_{i=1}^{N-1} sum_{j=i+1}^N f(A_i, A_j)
# where f(x, y) = (x + y) % M
# S = sum_{i<j} ((A_i + A_j) % M)
# We know (x + y) % M = (x + y) - floor((x + y) / M) * M
# S = sum_{i<j} (A_i + A_j) - sum_{i<j} (floor((A_i + A_j) / M) * M)
# S = sum_{i<j} (A_i + A_j) - M * sum_{i<j} floor((A_i + A_j) / M)

# The first part: sum_{i<j} (A_i + A_j)
# Each A_k appears in (N-1) pairs (with A_j where j != k).
# In the sum over i < j, each A_k appears in (N-1) pairs.
# For k=1, pairs are (1,2), (1,3), ..., (1,N) => N-1 pairs.
# For k=2, pairs are (1,2), (2,3), ..., (2,N). A_2 appears in (1,2) and (2,j) for j > 2. Total 1 + (N-2) = N-1 pairs.
# In general, A_k appears in pairs (i,k) with i < k (k-1 pairs) and (k,j) with j > k (N-k pairs). Total (k-1) + (N-k) = N-1 pairs.
# So, sum_{i<j} (A_i + A_j) = sum_{k=1}^N (N-1) * A_k = (N-1) * sum_{k=1}^N A_k
# Use Python's arbitrary precision integer type
sum_A = sum(A)
ans1 = (N - 1) * sum_A

# The second part: sum_{i<j} floor((A_i + A_j) / M)
# Since 1 <= A_i < M and 1 <= A_j < M, 2 <= A_i + A_j < 2M.
# So floor((A_i + A_j) / M) is either 0 or 1.
# floor((A_i + A_j) / M) = 1 if A_i + A_j >= M
# floor((A_i + A_j) / M) = 0 if A_i + A_j < M
# The sum sum_{i<j} floor((A_i + A_j) / M) is the count of pairs (i, j) with i < j such that A_i + A_j >= M.
count_ge_M = 0

# We can count pairs (i, j) with i < j and A_i + A_j >= M by iterating through i
# and for each A[i], counting A[j] with j > i such that A[j] >= M - A[i].
# An efficient way is to iterate through the array and use a Fenwick tree.
# When processing A[i], count A[j] with j < i such that A[j] >= M - A[i] (threshold).
# This is equivalent to counting pairs (j, i) with j < i such that A_j + A_i >= M.

# Get sorted distinct values from A to use as ranks for the Fenwick tree.
U = sorted(list(set(A)))
D = len(U) # Number of distinct values

# Initialize Fenwick tree with size D (for ranks 1 to D)
ft = FenwickTree(D)

# Iterate through A from left to right (i from 0 to N-1)
# When at index i, the Fenwick tree contains the counts of values A[0], ..., A[i-1]
for i in range(N):
    current_value = A[i]

    # We want to count A[j] (where j < i, already in FT) such that A[j] + current_value >= M
    # This is equivalent to A[j] >= M - current_value
    threshold = M - current_value

    # Find the number of elements in U that are >= threshold
    # bisect_left finds the 0-based index k in U such that U[k] is the first element >= threshold
    # All elements in U from index k onwards are >= threshold
    # These correspond to 1-based ranks k+1, k+2, ..., D

    # Find the 0-based index k in U where U[k] >= threshold
    k_0based = bisect_left(U, threshold)

    # Count elements in the FT with 1-based rank in [k_0based + 1, D]
    # The number of elements with rank <= k_0based is ft.query(k_0based).
    # The total number of elements in FT is ft.query(D).
    # The number of elements with rank > k_0based (i.e., rank >= k_0based + 1) is ft.query(D) - ft.query(k_0based).
    count_current_pairs = ft.query(D) - ft.query(k_0based)

    count_ge_M += count_current_pairs

    # Add current_value to the Fenwick tree for processing future elements A[i'] where i' > i
    # Find the 0-based rank of current_value in U
    current_rank_0based = bisect_left(U, current_value)
    # The 1-based rank is current_rank_0based + 1
    ft.update(current_rank_0based + 1, 1)

# Use Python's arbitrary precision integer type for the final calculation
ans2 = M * count_ge_M

# Final answer is ans1 - ans2
final_answer = ans1 - ans2

# Print the result
print(final_answer)