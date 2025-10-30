# YOUR CODE HERE
import sys

# Fenwick tree (BIT) implementation class
class FenwickTree:
    """
    A Fenwick tree (Binary Indexed Tree) implementation supporting point updates and prefix sum queries.
    It uses 1-based indexing internally for the tree array.
    The size parameter provided during initialization should be the maximum possible index (N in this problem).
    """
    def __init__(self, size):
        # Initialize the tree array with zeros. Size is N+1 for 1-based indexing.
        self.tree = [0] * (size + 1)
        self.size = size # Maximum index supported (N)

    def update(self, idx, val):
        """
        Adds `val` to the element at index `idx`.
        Index `idx` is assumed to be 1-based.
        """
        # Check if index is valid (should be 1 <= idx <= N)
        if not (1 <= idx <= self.size):
             # This case should ideally not happen if input is valid (permutation of 1..N)
             # For robustness, we can either raise an error or return.
             # Since P contains values 1..N, idx will be in range.
             return
        
        # Standard BIT update logic:
        # Propagate the update upwards in the tree structure.
        while idx <= self.size:
            self.tree[idx] += val
            # Move to the next index affected by this update.
            # This is done by adding the value of the least significant bit.
            idx += idx & (-idx)

    def query(self, idx):
        """
        Queries the cumulative sum of elements from index 1 up to `idx`.
        Index `idx` is assumed to be 1-based.
        """
        # Clamp idx to the valid range [0, self.size].
        # Querying up to idx=0 should yield 0.
        if idx > self.size:
             idx = self.size
        elif idx < 0:
             return 0 # Prefix sum up to 0 or negative index is 0.
        
        s = 0
        # Standard BIT query logic:
        # Sum values by moving downwards towards index 0.
        while idx > 0:
            s += self.tree[idx]
            # Move to the index representing the prefix excluding the current range.
            # This is done by subtracting the value of the least significant bit.
            idx -= idx & (-idx)
        return s

def solve():
    """
    Reads input, calculates the minimum cost to sort the permutation using the described operation,
    and prints the result.
    Uses a Fenwick tree to efficiently calculate costs based on an insertion sort-like process.
    """
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    # Initialize Fenwick tree. The tree will store counts based on the values (1 to N).
    ft = FenwickTree(N)

    total_cost = 0

    # Iterate through the permutation elements P_1, ..., P_N.
    # The loop variable `i` represents the 1-based index of the current element being processed.
    # This corresponds to the position in the array P, and is used in cost calculation.
    for i in range(1, N + 1):
        # Get the value `k` of the element at the current position `i`.
        # P is 0-indexed, so we access P[i-1].
        k = P[i-1] 

        # Calculate the number of elements already processed (those appearing before index `i` in P)
        # that have a value greater than `k`.
        # The BIT `ft` stores '1' at the index corresponding to the value of each processed element.
        # `ft.query(N)` gives the total count of elements processed so far (i-1 elements).
        # `ft.query(k)` gives the count of processed elements with value <= k.
        # Their difference `ft.query(N) - ft.query(k)` gives the count of processed elements with value > k.
        
        count_greater = ft.query(N) - ft.query(k)
        
        # `count_greater` represents the number of elements larger than `k` that are currently
        # positioned to the left of `k` (among the first `i` elements).
        # In an insertion sort like process, `k` would need to be swapped leftwards past these `count_greater` elements.
        # The swaps would occur successively at indices `i-1`, `i-2`, ..., down to `i - count_greater`.
        # Note that indices here refer to the 1-based indices used for cost calculation.
        # The cost for inserting element `k` is the sum of these swap indices.
        
        if count_greater > 0:
            # The sequence of swap indices forms an arithmetic progression.
            # The first swap index is `i - count_greater`.
            # The last swap index is `i - 1`.
            first_idx = i - count_greater
            last_idx = i - 1
            
            # Calculate the sum of this arithmetic series: (first + last) * num_terms / 2
            # The number of terms is `count_greater`.
            # Use integer division // since the result must be an integer.
            current_cost = (first_idx + last_idx) * count_greater // 2
            total_cost += current_cost

        # After processing element `k` and calculating its contribution to the cost,
        # mark `k` as processed by adding 1 to the count at index `k` in the Fenwick tree.
        # This signifies that an element with value `k` has been encountered.
        ft.update(k, 1)

    # Print the final computed total minimum cost.
    print(total_cost)

# Execute the solve function when the script is run
solve()