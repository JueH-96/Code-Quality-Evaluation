import sys

class FenwickTree:
    """
    A class for a Fenwick Tree (Binary Indexed Tree).
    It supports point updates and prefix sum queries in O(log n) time.
    The tree is implemented to be 1-indexed for simpler update/query logic.
    """
    def __init__(self, size):
        """
        Initializes a Fenwick Tree of a given size.
        The internal array has size + 1 to accommodate 1-based indexing.
        """
        self.tree = [0] * (size + 1)
        self.size = size

    def update(self, index, value):
        """
        Adds `value` to the element at `index`.
        `index` must be a 1-based integer.
        """
        while index <= self.size:
            self.tree[index] += value
            index += index & (-index)

    def query(self, index):
        """
        Calculates the prefix sum up to `index`.
        Returns the sum of elements from 1 to `index`.
        If index is 0, it correctly returns 0.
        """
        s = 0
        while index > 0:
            s += self.tree[index]
            index -= index & (-index)
        return s

# Use sys.stdin.readline for faster I/O in competitive programming
input = sys.stdin.readline

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        N_str = input()
        if not N_str: return
        N = int(N_str)
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        return

    # 1. Coordinate Compression
    # Map the potentially large values in A to a smaller range of ranks.
    unique_vals = sorted(list(set(A)))
    val_to_rank = {val: i + 1 for i, val in enumerate(unique_vals)}
    num_unique = len(unique_vals)

    # 2. Initialization of Fenwick Trees
    # count_bit will store counts of elements based on their rank.
    # sum_bit will store sums of original values based on their rank.
    count_bit = FenwickTree(num_unique)
    sum_bit = FenwickTree(num_unique)
    
    total_sum = 0

    # 3. Iterate through the array and calculate the total sum
    for current_val in A:
        # Get the 1-based rank of the current value.
        rank = val_to_rank[current_val]

        # Query for count and sum of elements seen so far with rank < current rank.
        # This corresponds to querying up to rank - 1.
        num_smaller = count_bit.query(rank - 1)
        sum_of_smaller = sum_bit.query(rank - 1)

        # Calculate and add the contribution of the current element:
        # contribution = sum_{i < j, A_i < A_j} (A_j - A_i)
        #            = A_j * count({i | i<j, A_i<A_j}) - sum({A_i | i<j, A_i<A_j})
        contribution = current_val * num_smaller - sum_of_smaller
        total_sum += contribution

        # Update the Fenwick trees with the current element's information.
        count_bit.update(rank, 1)
        sum_bit.update(rank, current_val)

    # 4. Print the final result
    print(total_sum)

solve()