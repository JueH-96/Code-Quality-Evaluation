import bisect

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    X = list(map(int, data[1::2]))
    Y = list(map(int, data[2::2]))
    
    balls = list(zip(X, Y))
    # Sort by X increasing, then Y increasing
    balls.sort()
    
    # Compress Y values
    sorted_Y = [b[1] for b in balls]
    unique_Y = sorted(sorted_Y)
    rank = {y: i+1 for i, y in enumerate(unique_Y)}  # 1-based indexing for Fenwick Tree
    
    # Fenwick Tree (1-based indexing)
    class FenwickTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 1)
        
        def update(self, idx, delta):
            while idx <= self.n:
                self.tree[idx] = (self.tree[idx] + delta) % MOD
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res = (res + self.tree[idx]) % MOD
                idx -= idx & -idx
            return res
    
    ft = FenwickTree(len(unique_Y)+2)  # len(unique_Y) is up to N (300)
    
    ans = 0
    for x, y in balls:
        r = rank[y]
        current = ft.query(r - 1) + 1  # sum of all ways before this y, plus one for the current ball alone
        ans = (ans + current) % MOD
        ft.update(r, current)
    
    print(ans % MOD)

if __name__ == '__main__':
    main()