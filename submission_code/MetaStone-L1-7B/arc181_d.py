import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    M = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    # Initialize Fenwick Tree
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
    inv_count = 0
    
    for a in A:
        k = a
        # Process the first k elements
        for i in range(k-1):
            if P[i] > P[i+1]:
                # Remove inversion (i, i+1)
                inv_count -= 1
                # Update Fenwick Tree
                ft.update(P[i+1])
                ft.update(P[i])
                # Now, check how many elements to the right of i are smaller than P[i]
                # The number of elements smaller than P[i] in [i+2, k]
                count_smaller = ft.query(P[i]-1) - ft.query(i+1)
                inv_count += count_smaller
                # Swap
                P[i], P[i+1] = P[i+1], P[i]
        print(inv_count)

if __name__ == '__main__':
    main()