# YOUR CODE HERE
import sys

class FenwickTree:
    """A class for Fenwick Tree (Binary Indexed Tree)."""
    def __init__(self, size):
        """
        Initializes a Fenwick Tree of a given size.
        All values are initially zero.
        """
        self.tree = [0] * (size + 1)
        self.size = size

    def add(self, i, x):
        """Adds x to the value at index i."""
        i += 1  # 1-based index
        while i <= self.size:
            self.tree[i] += x
            i += i & -i

    def prefix_sum(self, i):
        """Calculates the sum of values from index 0 to i."""
        if i < 0:
            return 0
        i += 1  # 1-based index
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

def solve():
    """
    Main function to solve the problem.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        
        N = int(input())
        # Read permutation and convert to 0-indexed values
        P = [int(x) - 1 for x in input().split()]
        M = int(input())
        if M > 0:
            A = [int(x) for x in input().split()]
        else:
            A = []

    except (IOError, ValueError):
        # Handle potential empty lines or formatting errors in input
        return

    # 1. Calculate the initial inversion number
    # We use a Fenwick tree for an efficient O(N log N) calculation.
    ft = FenwickTree(N)
    inversions = 0
    # Iterate from right to left. For each element P[i], count how many
    # elements to its right are smaller than it.
    for i in range(N - 1, -1, -1):
        inversions += ft.prefix_sum(P[i] - 1)
        ft.add(P[i], 1)
    
    current_P = list(P)
    
    # 2. Process operations sequentially
    # For each operation, we simulate a bubble sort pass and count swaps.
    # The new inversion count is the old count minus the number of swaps.
    for k in A:
        swaps = 0
        # Operation k: a single bubble sort pass on the prefix P[0...k-1]
        for i in range(k - 1):
            if current_P[i] > current_P[i+1]:
                current_P[i], current_P[i+1] = current_P[i+1], current_P[i]
                swaps += 1
        
        inversions -= swaps
        print(inversions)

if __name__ == "__main__":
    solve()