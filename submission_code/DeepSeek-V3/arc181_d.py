# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    P = list(map(int, data[idx:idx+N]))
    idx += N
    M = int(data[idx])
    idx += 1
    A = list(map(int, data[idx:idx+M]))
    
    # Initialize the inversion count
    inv = 0
    # Precompute the inversion count for the initial permutation
    # Using a BIT (Fenwick Tree) to count inversions
    class BIT:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 2)
        
        def update(self, idx, delta=1):
            while idx <= self.n:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    
    bit = BIT(N)
    for num in reversed(P):
        inv += bit.query(num - 1)
        bit.update(num)
    
    # Now, process the operations
    # We need to simulate the operations and update the inversion count
    # Since A is non-decreasing, we can process them in order
    # For each operation k, we perform the swaps and update the inversion count
    # To efficiently manage the swaps, we need to keep track of the positions of each element
    # We can use a list to represent the permutation and a dictionary to map elements to their positions
    
    # Initialize the position map
    pos = {x: i for i, x in enumerate(P)}
    
    for a in A:
        # Perform operation a
        # For i from 1 to a-1, if P[i-1] > P[i], swap them
        # Since we are processing in order, we can iterate and swap
        # Also, update the inversion count accordingly
        for i in range(a-1):
            if P[i] > P[i+1]:
                # Swap P[i] and P[i+1]
                # Update the inversion count
                # Before swap, P[i] > P[i+1], so the inversion count decreases by 1
                inv -= 1
                # Swap the elements
                P[i], P[i+1] = P[i+1], P[i]
                # Update the positions
                pos[P[i]] = i
                pos[P[i+1]] = i+1
        # After performing the operation, print the current inversion count
        print(inv)

if __name__ == "__main__":
    main()