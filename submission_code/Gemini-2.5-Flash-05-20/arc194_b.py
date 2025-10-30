import sys

class FenwickTree:
    def __init__(self, size):
        # The Fenwick Tree is 1-indexed internally.
        # `size` refers to the maximum value (or maximum index) it can store.
        self.tree = [0] * (size + 1)
        self.size = size

    def update(self, index, delta):
        # Converts 0-indexed input `index` to 1-indexed for internal use.
        index += 1
        while index <= self.size:
            self.tree[index] += delta
            index += index & (-index)

    def query(self, index):
        # Converts 0-indexed input `index` (query sum up to this index inclusive)
        # to 1-indexed for internal use.
        index += 1
        s = 0
        while index > 0:
            s += self.tree[index]
            index -= index & (-index)
        return s

def solve():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    total_cost = 0
    
    # We use a Fenwick Tree (BIT) to efficiently count elements.
    # The values in P are from 1 to N. We map them to 0 to N-1 for BIT indices.
    # The BIT `ft` will store counts of values that have been processed so far.
    ft = FenwickTree(N) 
    
    # The problem asks for the minimum total cost to sort P.
    # The sample explanation for P=(3,2,1) directly corresponds to a standard bubble sort execution,
    # and it states that 4 is the minimum cost. This strongly suggests that standard bubble sort
    # sequence of operations provides the minimum cost in this problem.
    # The cost of swapping P_i and P_{i+1} is `i` (1-indexed). So for 0-indexed `P[k]` and `P[k+1]`, the cost is `k+1`.

    # The total cost of bubble sort is the sum of costs for all swaps performed.
    # A property of bubble sort is that the number of times `P[k]` and `P[k+1]` are swapped
    # across all passes is equal to the number of elements `X` that are at `P_j` where `j > k`
    # (i.e., to the right of `k`) and `X < P[k]` (i.e., `X` is smaller than the element at `P[k]`).
    # In other words, for each element `P[k]`, we sum `(k+1)` for every inversion it forms with an element to its right.

    # We iterate through the array `P` from right to left (from `N-1` down to `0`).
    # This allows us to use the Fenwick Tree to efficiently query for elements to the right.
    for i in range(N - 1, -1, -1):
        val = P[i] # Current value at index `i`. `val` is between 1 and N.
        
        # `ft.query(val - 1)`:
        # This queries the count of elements already added to the Fenwick Tree (i.e., elements
        # that are to the right of the current index `i` in the original array `P`)
        # that are strictly less than `val`.
        # `val - 1` is used to convert `val` (1 to N) to a 0-indexed value for BIT query.
        num_smaller_on_right = ft.query(val - 1)
        
        # Each `num_smaller_on_right` element signifies an inversion (val, x) where x < val and x is to the right of val.
        # For each such inversion, `val` must effectively "bubble left" past `x`. This contributes to a swap at index `i`.
        # The cost of such a swap at 0-indexed `i` is `i+1`.
        total_cost += num_smaller_on_right * (i + 1)
        
        # Add the current value `val` to the Fenwick Tree, marking it as "processed" (i.e., now to the left).
        # `val - 1` converts `val` (1 to N) to a 0-indexed index for BIT update.
        ft.update(val - 1, 1)

    sys.stdout.write(str(total_cost) + '
')

solve()