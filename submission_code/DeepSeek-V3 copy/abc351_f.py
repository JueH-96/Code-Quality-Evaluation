def calculate_expression(N, A):
    # Create a list of tuples (A_i, i) to keep track of original indices
    indexed_A = [(A[i], i) for i in range(N)]
    # Sort the list based on the value A_i
    indexed_A.sort()
    # Initialize a Binary Indexed Tree (Fenwick Tree) to count the number of elements less than a certain value
    class FenwickTree:
        def __init__(self, size):
            self.N = size + 2
            self.tree = [0] * (self.N)
        
        def update(self, index, value):
            while index < self.N:
                self.tree[index] += value
                index += index & -index
        
        def query(self, index):
            res = 0
            while index > 0:
                res += self.tree[index]
                index -= index & -index
            return res
    
    # Initialize the Fenwick Tree
    ft = FenwickTree(N)
    total = 0
    # Iterate through the sorted list
    for val, idx in indexed_A:
        # Query the number of elements before the current index that are less than the current value
        count = ft.query(idx)
        # Add to the total the number of such elements multiplied by the current value
        total += count * val
        # Update the Fenwick Tree with the current index
        ft.update(idx + 1, 1)
    # Now, calculate the sum of all A_i * (number of elements after it that are greater than it)
    # To do this, we can iterate in reverse order
    ft_rev = FenwickTree(N)
    total_rev = 0
    for val, idx in reversed(indexed_A):
        # Query the number of elements after the current index that are greater than the current value
        count = ft_rev.query(N) - ft_rev.query(idx + 1)
        # Add to the total the number of such elements multiplied by the current value
        total_rev += count * val
        # Update the Fenwick Tree with the current index
        ft_rev.update(idx + 1, 1)
    # The final result is total_rev - total
    # Because total_rev is the sum of A_j * (number of i < j where A_i < A_j)
    # And total is the sum of A_i * (number of j > i where A_j > A_i)
    # So total_rev - total is the sum of (A_j - A_i) for all i < j where A_j > A_i
    result = total_rev - total
    print(result)

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))
# Calculate and print the result
calculate_expression(N, A)