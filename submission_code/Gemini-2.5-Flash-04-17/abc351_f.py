import sys

# Binary Indexed Tree (Fenwick Tree) Implementation
class BIT:
    def __init__(self, size):
        # size is 1 + maximum index we will use.
        # If ranks are 0 to k-1, maximum index in BIT is k. So size is k+1.
        self.size = size
        self.tree = [0] * size # Using 0-based array, but BIT operations use 1-based logic on indices

    def update(self, index, value):
        # index is 1-based for the BIT operation
        # e.g., rank r (0-based) maps to index r+1
        while index < self.size:
            self.tree[index] += value
            index += index & (-index)

    def query(self, index):
        # index is 1-based for the BIT operation
        # query sum from index 1 up to 'index'
        sum_val = 0
        while index > 0:
            sum_val += self.tree[index]
            index -= index & (-index)
        return sum_val

# Main logic
def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Coordinate Compression
    # Get distinct values and sort them
    distinct_values = sorted(list(set(A)))
    # Create a map from value to its rank (0-based)
    value_to_rank = {v: i for i, v in enumerate(distinct_values)}
    k = len(distinct_values) # Number of distinct ranks

    # Initialize two BITs: one for counts, one for sums
    # Size k+1 to handle ranks 0 to k-1 which map to BIT indices 1 to k
    bit_count = BIT(k + 1) # Stores count of elements seen so far for each rank category
    bit_sum = BIT(k + 1)   # Stores sum of values seen so far for each rank category

    total_sum = 0

    # Iterate through the input sequence A from left to right
    # For each element A[j], we calculate its contribution to the total sum
    # from pairs (i, j) where i < j and A[i] < A[j]
    for j in range(N):
        v = A[j]
        # Get the 0-based rank of the current value A[j]
        r = value_to_rank[v]
        # The corresponding 1-based index in the BIT is r + 1

        # Query the BITs for elements A[i] with i < j such that A[i] < A[j]
        # A[i] < A[j] means A[i] has a rank strictly less than the rank of A[j].
        # If rank of A[j] is r, we need ranks 0, 1, ..., r-1.
        # These ranks map to BIT indices 1, 2, ..., r.
        # So, we query the BIT up to index r.
        # If r is 0, there are no ranks less than 0, query(0) correctly returns 0.
        count_smaller = bit_count.query(r) # Count of A[i] with i<j and A[i] < A[j]
        sum_smaller = bit_sum.query(r)     # Sum of A[i] with i<j and A[i] < A[j]

        # The contribution for the current element A[j] from all previous smaller elements A[i] is:
        # sum(A[j] - A[i] for i < j and A[i] < A[j])
        # = sum(A[j]) - sum(A[i]) for i < j and A[i] < A[j]
        # = A[j] * (count of such A[i]) - (sum of such A[i])
        total_sum += v * count_smaller - sum_smaller

        # After processing A[j], update the BITs to include A[j]
        # A[j] has rank r (0-based), which corresponds to BIT index r + 1
        bit_count.update(r + 1, 1) # Add 1 to the count for rank r
        bit_sum.update(r + 1, v)   # Add the value A[j] to the sum for rank r

    print(total_sum)

solve()