import sys
from collections import defaultdict

MOD = 998244353

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    S = defaultdict(list)
    for _ in range(M):
        L, R, X = map(int, sys.stdin.readline().split())
        S[X].append((L, R))
    
    # Preprocess S_p for each p
    S_p = [[] for _ in range(N+1)]  # S_p[p] contains list of (L, R)
    for X in S:
        for L, R in S[X]:
            if L <= X <= R:
                S_p[X].append((L, R))
    
    # Segment tree for range maximum query and point update
    size = 1
    while size < N:
        size <<= 1
    max_tree = [0] * (2 * size)
    
    def update(pos, value):
        pos += size - 1
        if max_tree[pos] >= value:
            return
        max_tree[pos] = value
        while pos > 1:
            pos >>= 1
            new_val = max(max_tree[2*pos], max_tree[2*pos+1])
            if max_tree[pos] == new_val:
                break
            max_tree[pos] = new_val
    
    def query(l, r):
        res = 0
        l += size - 1
        r += size - 1
        while l <= r:
            if l % 2 == 1:
                res = max(res, max_tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, max_tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res
    
    placed = set()
    result = 1
    
    for k in reversed(range(1, N+1)):
        valid_p = 0
        for p in range(1, N+1):
            if p in placed:
                continue
            valid = True
            for L, R in S_p[p]:
                current_max = query(L, R)
                if current_max == 0:
                    valid = False
                    break
            if valid:
                valid_p += 1
        if valid_p == 0:
            print(0)
            return
        result = (result * valid_p) % MOD
        # Choose any valid p (we don't care which one, as the count is correct)
        # Find the first valid p
        for p in range(1, N+1):
            if p in placed:
                continue
            valid = True
            for L, R in S_p[p]:
                current_max = query(L, R)
                if current_max == 0:
                    valid = False
                    break
            if valid:
                placed.add(p)
                update(p, k)
                break
    
    print(result % MOD)

if __name__ == '__main__':
    main()