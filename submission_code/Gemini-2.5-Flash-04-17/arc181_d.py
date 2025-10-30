import sys

# Fenwick Tree (Binary Indexed Tree)
class FenwickTree:
    def __init__(self, size):
        # size + 1 because values are 1-based, tree uses 1-based index
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, idx, delta):
        # Update value at idx (1-based)
        # Traverse ancestors
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & (-idx) # Move to parent node

    def query(self, idx):
        # Get sum up to idx (1-based)
        # Traverse ancestors
        _sum = 0
        while idx > 0:
            _sum += self.tree[idx]
            idx -= idx & (-idx) # Move to parent node
        return _sum

# Function to calculate initial inversion number using Fenwick Tree
def calculate_inversions(p, n):
    bit = FenwickTree(n)
    inversions = 0
    # Iterate from right to left (N-1 down to 0)
    # For P[i], count elements < P[i] that are to its right.
    for i in range(n - 1, -1, -1):
        # Values in P are 1-based, use them directly as BIT indices
        # Query sum in range [1, P[i]-1]
        inversions += bit.query(p[i] - 1)
        # Add P[i] to the BIT
        bit.update(p[i], 1)
    return inversions

def solve():
    n = int(sys.stdin.readline())
    # Read permutation, convert to 0-indexed list. Values are 1..N.
    p = list(map(int, sys.stdin.readline().split())) 
    m = int(sys.stdin.readline())
    # Read operations sequence, values are 2..N.
    a = list(map(int, sys.stdin.readline().split())) 

    # Calculate initial inversion number
    current_inversions = calculate_inversions(p, n)

    # Use a mutable list for the permutation
    P_list = list(p)

    # Simulate operations
    for k in a:
        # Operation k: bubble sort pass on P[0...k-1]
        # As per problem description "for i=1,2,...,k-1 in this order, if P_i > P_{i+1}, swap" (1-based)
        # This translates to 0-based indices: for i = 0, 1, ..., k-2, if P[i] > P[i+1], swap P[i], P[i+1].
        for i in range(k - 1):
            # Compare P_list[i] and P_list[i+1]
            if P_list[i] > P_list[i+1]:
                # Swap in P_list
                P_list[i], P_list[i+1] = P_list[i+1], P_list[i]

                # An adjacent swap of elements in inversion order decreases
                # the total inversion count by exactly 1.
                current_inversions -= 1

        # Print inversion count after the operation
        print(current_inversions)

solve()