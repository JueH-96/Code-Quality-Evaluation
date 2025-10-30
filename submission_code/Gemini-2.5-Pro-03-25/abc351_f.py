# YOUR CODE HERE
import sys

# Fenwick Tree (BIT) implementation class
# Uses 1-based indexing for tree operations
class FenwickTree:
    def __init__(self, size):
        """Initializes a Fenwick tree of given size."""
        # The internal array uses 1-based indexing. 
        # It needs to be of size M+1 to support indices 1 to M.
        self.bit = [0] * (size + 1) 
        self.size = size # Stores the maximum rank/index M

    def update(self, idx, delta):
        """Adds delta to the element at index idx."""
        # idx must be >= 1 and <= self.size for valid updates
        # The loop updates the node at idx and all its ancestors in the Fenwick tree structure.
        while idx <= self.size:
            self.bit[idx] += delta
            # Move to the next index that represents a larger range covering idx
            # The expression `idx & (-idx)` isolates the least significant bit of idx.
            # Adding this value moves to the parent node in the update path.
            idx += idx & (-idx) 

    def query(self, idx):
        """Computes the prefix sum up to index idx (sum of elements from 1 to idx)."""
        # Querying index 0 or any negative index should return 0, as there are no elements.
        s = 0
        # The loop sums values from nodes corresponding to ranges that cover the prefix [1, idx].
        while idx > 0:
            s += self.bit[idx]
            # Move to the index representing the end of the range just before the least significant bit range of idx.
            # Subtracting `idx & (-idx)` moves to the parent node in the query path.
            idx -= idx & (-idx)
        return s

def solve():
    """Reads input, computes the required sum using Fenwick trees, and prints the result."""
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Coordinate Compression
    # Step 1: Find all unique values in A and sort them.
    unique_A = sorted(list(set(A)))
    # M is the number of unique values. This will be the size of our Fenwick Trees' effective range.
    M = len(unique_A) 
    
    # Step 2: Create a mapping from each unique value to its rank.
    # We use 1-based ranks (1 to M) because Fenwick trees are typically implemented with 1-based indexing.
    val_to_rank = {val: i + 1 for i, val in enumerate(unique_A)} 
    
    # Initialize Fenwick Trees
    # count_bit: Stores counts of elements encountered so far, indexed by their rank.
    #            `count_bit.query(r)` gives the total count of elements with rank <= r.
    count_bit = FenwickTree(M)
    # sum_bit: Stores sums of values of elements encountered so far, indexed by their rank.
    #          `sum_bit.query(r)` gives the total sum of values of elements with rank <= r.
    sum_bit = FenwickTree(M)

    total_sum = 0
    
    # Iterate through the array A using 0-based index k (from 0 to N-1)
    for k in range(N):
        current_val = A[k]
        # Get the 1-based rank of the current value using the precomputed mapping.
        current_rank = val_to_rank[current_val] 
        
        # Calculate the contribution of pairs (i, k) where i < k and A[k] > A[i].
        # The expression requires summing (A[k] - A[i]) for all such pairs.
        # This sum can be rewritten as: Sum(A[k]) - Sum(A[i]) over the valid pairs.
        # Sum(A[k]) = A[k] * Count(i < k such that A[i] < A[k])
        # Sum(A[i]) = Sum(A[i] for i < k such that A[i] < A[k])
        
        # Query count_bit for the number of elements processed so far (indices i < k) 
        # whose value A[i] is strictly less than current_val (A[k]).
        # These elements have ranks from 1 up to current_rank - 1.
        # `count_bit.query(current_rank - 1)` computes the sum of counts for ranks 1..current_rank-1.
        count_less = count_bit.query(current_rank - 1)
        
        # Query sum_bit for the sum of values A[i] for elements processed so far (indices i < k) 
        # whose value A[i] is strictly less than current_val (A[k]).
        # Similarly, these elements have ranks from 1 up to current_rank - 1.
        sum_less = sum_bit.query(current_rank - 1)
        
        # Calculate the contribution for the current element A[k].
        contribution = current_val * count_less - sum_less
        # Add this contribution to the total sum.
        total_sum += contribution
        
        # Update the Fenwick Trees with the information of the current element A[k].
        # Increment the count at its rank position by 1.
        count_bit.update(current_rank, 1)
        # Add its value to the sum at its rank position.
        sum_bit.update(current_rank, current_val)

    # Print the final accumulated sum.
    # The sum is guaranteed to fit within a 64-bit integer per problem statement. Python handles large integers automatically.
    print(total_sum)

# Execute the main function to solve the problem
solve()