import sys
from bisect import bisect_left

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    P = list(map(int, input[ptr:ptr+N]))
    ptr += N
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+M]))
    ptr += M

    # Compute initial inversion count using Fenwick Tree
    class FenwickTree:
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
        
        def range_query(self, l, r):
            return self.query(r) - self.query(l-1)
    
    ft = FenwickTree(N)
    initial_inversion = 0
    for i in reversed(range(N)):
        val = P[i]
        initial_inversion += ft.query(val-1)
        ft.update(val)
    
    # Preprocess for each k, the position of the maximum in the first k elements
    pos = [0] * (N+2)
    current_max_pos = 0
    max_val = P[0]
    pos[1] = 0
    for i in range(1, N):
        if P[i] > max_val:
            max_val = P[i]
            current_max_pos = i
        pos[i+1] = current_max_pos

    # Process each operation
    current_inversion = initial_inversion
    prev_k = 0
    for k in A:
        k_idx = k
        if k < 1:
            print(current_inversion)
            continue
        m = pos[k_idx]  # 0-based position of max in first k elements (0-based)
        swap_count = m  # elements to the left of m (0-based) is m elements (indices 0 to m-1)
        current_inversion -= swap_count
        print(current_inversion)
        prev_k = k

if __name__ == "__main__":
    main()