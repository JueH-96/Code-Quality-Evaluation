import sys

class FenwickTree:
    def __init__(self, size):
        self.tree = [0] * (size + 1)
        self.size = size

    def update(self, idx, delta_val): # 0-indexed idx
        idx += 1 # Convert to 1-indexed for BIT
        while idx <= self.size:
            self.tree[idx] += delta_val
            idx += idx & (-idx)
            
    def query(self, idx): # 0-indexed idx, query sum for [0...idx]
        idx += 1 # Convert to 1-indexed for BIT
        s = 0
        while idx > 0:
            s += self.tree[idx]
            idx -= idx & (-idx)
        return s

def solve():
    N = int(sys.stdin.readline())
    P_list = list(map(int, sys.stdin.readline().split()))

    # Store 0-indexed positions of values (1 to N)
    pos_of_val = [-1] * (N + 1) 
    for i in range(N):
        pos_of_val[P_list[i]] = i

    # Fenwick tree for sum of (original_0_idx + 1) which is the 1-based cost
    fen_indices_sum = FenwickTree(N)

    # Initialize Fenwick tree: for each element P[i] at original_0_idx=i,
    # its cost contribution if it's part of a swap is (i+1).
    for i in range(N):
        fen_indices_sum.update(i, i + 1)

    total_cost = 0
    for val_to_place in range(1, N + 1):
        # original_idx is the 0-indexed position of val_to_place in the initial P
        original_idx = pos_of_val[val_to_place]
        
        # Query sum of costs for active elements strictly to the left of original_idx.
        # These are the costs of swaps val_to_place will make as it moves left.
        # query(k) sums from index 0 to k. So original_idx-1 for elements to the left.
        if original_idx == 0:
            cost_for_this_val = 0
        else:
            cost_for_this_val = fen_indices_sum.query(original_idx - 1)
        
        total_cost += cost_for_this_val
        
        # Mark element at original_idx as "placed" by removing its cost contribution.
        # The cost factor (original_idx + 1) is removed.
        fen_indices_sum.update(original_idx, -(original_idx + 1))
        
    print(total_cost)

solve()