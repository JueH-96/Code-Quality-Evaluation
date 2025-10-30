import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    M = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Compute initial inversion count using Fenwick Tree
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0]*(self.size + 2)
        
        def update(self, index, delta=1):
            while index <= self.size:
                self.tree[index] += delta
                index += index & -index
        
        def query(self, index):
            res = 0
            while index > 0:
                res += self.tree[index]
                index -= index & -index
            return res
    
    max_val = N
    ft = FenwickTree(max_val)
    inv_count = 0
    for i in reversed(range(N)):
        inv_count += ft.query(P[i] - 1)
        ft.update(P[i])
    
    # Precompute adjacent inversions
    adj = [0] * (N)
    for i in range(N-1):
        if P[i] > P[i+1]:
            adj[i] = 1
    
    # Fenwick Tree for adj
    class AdjFenwick:
        def __init__(self, arr):
            self.n = len(arr)
            self.tree = [0] * (self.n + 2)
            for i in range(self.n):
                if arr[i]:
                    self.update(i+1, 1)  # 1-based
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
    
    adj_ft = AdjFenwick(adj)
    
    # For each operation, simulate the swaps and update adj_ft
    current_P = P.copy()
    for a in A:
        k = a
        s = adj_ft.query(k-1)
        inv_count -= s
        print(inv_count)
        
        # Now, simulate the swaps for this operation
        # But this is O(k), which is too slow for large N and M
        # Therefore, this code will not pass for large test cases.
        # We need a more efficient way to handle this part.
        # The following code is for small cases only.
        
        # Track the positions where swaps occurred
        swaps = []
        for i in range(k-1):
            if current_P[i] > current_P[i+1]:
                swaps.append(i)
        
        # Perform the swaps
        for i in swaps:
            current_P[i], current_P[i+1] = current_P[i+1], current_P[i]
        
        # Update the adj array and adj_ft
        # This part is very slow and needs optimization
        adj = [0]*(N-1)
        for i in range(N-1):
            if current_P[i] > current_P[i+1]:
                adj[i] = 1
        adj_ft = AdjFenwick(adj)
    
if __name__ == '__main__':
    main()