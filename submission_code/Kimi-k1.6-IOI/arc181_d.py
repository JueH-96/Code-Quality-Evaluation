import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
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
        if l > r:
            return 0
        return self.query(r) - self.query(l - 1)

def main():
    import sys
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
    sorted_P = sorted(P)
    rank = {v: i+1 for i, v in enumerate(sorted_P)}
    ft = FenwickTree(N)
    initial_inv = 0
    for i in range(N-1, -1, -1):
        initial_inv += ft.query(rank[P[i]] - 1)
        ft.update(rank[P[i]])
    
    # Now process each operation and maintain the inversion count
    current = 0
    ans = []
    processed = [False] * N
    ft = FenwickTree(N)
    total_swaps = 0
    for k in A:
        k -= 1  # Convert to 0-based index
        if k <= current:
            continue
        for i in range(current + 1, k):
            if not processed[i]:
                # Number of elements > P[i] already inserted
                r = rank[P[i]]
                swaps = ft.query(ft.n) - ft.query(r)
                total_swaps += swaps
                ft.update(r)
                processed[i] = True
        current = k
        ans.append(initial_inv - total_swaps)
    
    for cnt in ans:
        print(cnt)

if __name__ == "__main__":
    main()