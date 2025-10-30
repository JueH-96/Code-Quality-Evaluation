import sys

# Use sys.stdin.readline for faster input reading, especially for large N
input = sys.stdin.readline

class FenwickTree:
    """
    A Fenwick Tree (Binary Indexed Tree) implementation for sum queries and point updates.
    The tree supports 1-based indexing.
    """
    def __init__(self, size):
        # size represents the maximum possible index (value) that can be stored.
        # The internal tree array is of size (size + 1) to use 1-based indexing.
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, idx, delta):
        """
        Adds 'delta' to the element logically stored at index 'idx'.
        idx: The 1-based index to update. Must be in the range [1, self.size].
        delta: The value to add.
        """
        # Propagate the update up the tree structure.
        # The loop continues as long as the current index is within the tree bounds.
        while idx <= self.size:
            self.tree[idx] += delta
            # Move to the parent node responsible for a wider range containing 'idx'.
            # This is done by adding the least significant bit of 'idx'.
            idx += idx & (-idx)

    def query(self, idx):
        """
        Computes the prefix sum of elements from index 1 up to 'idx' (inclusive).
        idx: The 1-based index up to which the sum is calculated.
        If idx is less than 1, the sum is 0.
        """
        # Sum up the contributions from relevant nodes in the tree structure.
        total_sum = 0
        # The loop continues as long as the current index is valid (greater than 0).
        # If idx > self.size, the loop will naturally stop when idx becomes <= self.size
        # after some steps, effectively calculating the sum up to min(idx, self.size).
        while idx > 0:
            total_sum += self.tree[idx]
            # Move to the parent node that contributes to the sum up to the current 'idx'.
            # This is done by subtracting the least significant bit of 'idx'.
            idx -= idx & (-idx)
        return total_sum

    # query_range(l, r) is not explicitly needed for this problem, but is a common BIT operation.
    # def query_range(self, l, r):
    #     """
    #     Computes the sum of elements from index 'l' to 'r' (inclusive).
    #     l, r: 1-based indices for the start and end of the range.
    #     """
    #     if l > r:
    #         return 0
    #     return self.query(r) - self.query(l - 1)

# Read the input size N.
N = int(input())
# Read the array A. Split the line into integers and store them in a list.
A = list(map(int, input().split()))

# The problem constraints state that 1 <= A_i <= 10^6.
# We need a data structure that can handle values (which are used as indices in the BIT)
# up to 10^6. So, the size of our Fenwick tree should be at least 10^6.
MAX_VAL = 1000000

# Initialize the Fenwick tree. This tree will store sums of the values present in A.
# The size is set to MAX_VAL to cover the entire possible range of A_i values.
sum_bit = FenwickTree(MAX_VAL)

# Calculate the total sum of all elements in the input array A.
# This is a necessary component for efficiently calculating the sum of elements
# *greater than* a given A_i.
total_sum = sum(A)

# Populate the Fenwick tree with the elements of A.
# We iterate through each element 'a' in the input array A.
# For each element 'a', we update the Fenwick tree at the index corresponding to
# the value 'a' by adding the value 'a' itself.
# After this loop completes, calling sum_bit.query(x) will return the sum of all
# elements in the original array A whose values are less than or equal to x.
for a in A:
    # The value 'a' is used as the 1-based index in the Fenwick tree,
    # and the value 'a' is also the delta (the amount being added at this index).
    # Since 1 <= a <= 10^6, 'a' will always be a valid index for our Fenwick tree.
    sum_bit.update(a, a)

# Calculate the desired sum for each element A_i in the original sequence A.
results = []
# Iterate through the input array A in its original order.
for a_i in A:
    # Calculate the sum of all elements in A that are less than or equal to the current element a_i.
    # This is efficiently obtained using the query operation on the populated Fenwick tree.
    sum_le_ai = sum_bit.query(a_i)
    
    # The sum of elements strictly greater than a_i is equal to the total sum of all elements
    # minus the sum of elements that are less than or equal to a_i.
    # Example: Sum(> x) = Sum(Total) - Sum(<= x).
    sum_greater_ai = total_sum - sum_le_ai
    
    # Store the calculated result for this element A_i.
    results.append(sum_greater_ai)

# Print the final results.
# The '*' operator unpacks the 'results' list, passing its elements as separate arguments
# to the print function. By default, print separates arguments with a space.
print(*results)