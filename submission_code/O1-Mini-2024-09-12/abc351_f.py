import sys, bisect

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    N,*rest = map(int, sys.stdin.read().split())
    A = rest[:N]
    vals = sorted(list(set(A)))
    mapping = {v:i+1 for i,v in enumerate(vals)}
    M = len(vals)+2

    class Fenwick:
        def __init__(self, size):
            self.N = size
            self.tree = [0]*(self.N+2)
        def update(self, idx, val):
            while idx <= self.N:
                self.tree[idx] += val
                idx += idx & -idx
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res

    ft_count = Fenwick(M)
    ft_sum = Fenwick(M)
    ans = 0
    for a in A:
        idx = mapping[a]-1
        cnt = ft_count.query(idx)
        s = ft_sum.query(idx)
        ans += a*cnt - s
        ft_count.update(mapping[a],1)
        ft_sum.update(mapping[a],a)
    print(ans)

if __name__ == "__main__":
    main()