import sys
import bisect # Though not directly used for coord_map lookup, useful for general coordinate compression

# FenwickTree (Binary Indexed Tree) implementation
class FenwickTree:
    def __init__(self, size):
        # The tree array is 1-indexed internally.
        # `size` refers to the number of distinct elements or maximum coordinate value (0 to size-1).
        # So, the internal array needs to be of size `size + 1` to accommodate 1-based indexing
        # for elements from 1 to `size`.
        self.tree = [0] * (size + 1)
        self.size = size # This is the effective range [0, self.size-1] for compressed coordinates

    def update(self, index, delta):
        # Convert 0-based index (compressed coordinate) to 1-based for the Fenwick tree
        index += 1
        while index <= self.size:
            self.tree[index] += delta
            # Move to the next index to update (parent node in BIT structure)
            index += index & (-index)

    def query(self, index):
        # Convert 0-based index (compressed coordinate) to 1-based for the Fenwick tree
        # Returns the prefix sum from 0 up to `index` (inclusive)
        if index < 0: # If querying for elements before the first valid coordinate
            return 0
        
        index += 1
        s = 0
        while index > 0:
            s += self.tree[index]
            # Move to the parent index for sum calculation
            index -= index & (-index)
        return s

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # --- Coordinate Compression ---
    # 1. Get all unique values from array A
    unique_A_values = sorted(list(set(A)))
    
    # 2. Create a mapping from original value to its compressed coordinate (0-indexed rank)
    # This allows O(1) lookup of coordinates for each A[i]
    coord_map = {val: i for i, val in enumerate(unique_A_values)}
    
    # 3. Determine the size for Fenwick trees. It's the number of unique values.
    # If there are M unique values, their compressed coordinates will range from 0 to M-1.
    bit_array_size = len(unique_A_values)

    # --- Fenwick Tree Initialization ---
    # Initialize two Fenwick trees: one for counts and one for sums.
    # bit_count[coord] will store how many times elements with that coord have been seen.
    # bit_sum[coord] will store the sum of original values of elements with that coord.
    bit_count = FenwickTree(bit_array_size)
    bit_sum = FenwickTree(bit_array_size)

    total_expression_sum = 0

    # --- Iterate through the array and calculate sum ---
    # We iterate through the array A from left to right. For each A[k] (current_val),
    # we consider it as A_j in the sum (sum_{i<j} max(A_j - A_i, 0)).
    # We need to find all A[i] (where i < k) such that A[i] < A[k],
    # and add (A[k] - A[i]) to the total sum.
    for k in range(N):
        current_val = A[k]
        
        # Get the compressed coordinate for the current value.
        current_coord = coord_map[current_val]

        # Query Fenwick trees for elements seen so far (A[i] where i < k)
        # that are strictly smaller than current_val.
        # This means querying for elements whose compressed coordinate is less than current_coord.
        # So we query the Fenwick trees up to index (current_coord - 1).
        count_smaller_elements = bit_count.query(current_coord - 1)
        sum_of_smaller_elements = bit_sum.query(current_coord - 1)
        
        # Add the contribution of current_val to the total sum.
        # For each A[i] where i < k and A[i] < current_val (A[k]):
        # The term (A[k] - A[i]) is added.
        # This can be calculated as: (count_smaller_elements * A[k]) - sum_of_smaller_elements.
        total_expression_sum += (count_smaller_elements * current_val) - sum_of_smaller_elements

        # After processing current_val, update the Fenwick trees to include it.
        # This makes current_val available for future queries (when A[j] where j > k is processed).
        bit_count.update(current_coord, 1) # Increment count for current_coord
        bit_sum.update(current_coord, current_val) # Add current_val to sum for current_coord

    # Print the final calculated sum.
    print(total_expression_sum)

# Call the solve function to run the program.
solve()