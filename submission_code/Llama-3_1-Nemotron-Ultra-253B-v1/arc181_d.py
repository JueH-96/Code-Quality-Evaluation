import sys
from sys import stdin

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(stdin.readline())
    P = list(map(int, stdin.readline().split()))
    M = int(stdin.readline())
    A = list(map(int, stdin.readline().split()))
    
    # Compute initial inversion count
    inv_count = 0
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
    
    ft = FenwickTree(N)
    for i in reversed(range(N)):
        inv_count += ft.query(P[i] - 1)
        ft.update(P[i])
    
    # Simulate each operation and compute swaps
    current_P = P.copy()
    for k in A:
        swaps = 0
        for i in range(k-1):
            if current_P[i] > current_P[i+1]:
                current_P[i], current_P[i+1] = current_P[i+1], current_P[i]
                swaps += 1
        inv_count -= swaps
        print(inv_count)

if __name__ == '__main__':
    main()