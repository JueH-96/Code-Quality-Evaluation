MOD = 998244353

class SegmentTree:
    def __init__(self, size):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.size = size
        self.tree = [0] * (2 * self.n)
    
    def update(self, pos, value):
        pos += self.n - 1  # assuming 0-based input
        if self.tree[pos] >= value:
            return
        self.tree[pos] = value
        while pos > 1:
            pos >>= 1
            new_val = max(self.tree[2 * pos], self.tree[2 * pos + 1])
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
    
    def range_max(self, l, r):
        # [l, r) 0-based, closed interval for l, open for r
        res = 0
        l += self.n
        r += self.n
        while l < r:
            if l % 2:
                res = max(res, self.tree[l])
                l += 1
            if r % 2:
                r -= 1
                res = max(res, self.tree[r])
            l >>= 1
            r >>= 1
        return res

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    operations = []
    for _ in range(Q):
        P = int(data[idx])
        idx += 1
        V = int(data[idx])
        idx += 1
        operations.append((P, V))
    
    prefix_tree = SegmentTree(N)
    suffix_tree = SegmentTree(N)
    
    dp = 1
    for P, V in operations:
        # Check prefix option
        valid_prefix = False
        # Query prefix tree [0, P-1] (assuming 0-based)
        max_pre = prefix_tree.range_max(0, P)
        max_suf = suffix_tree.range_max(0, P)
        current_max = max(max_pre, max_suf)
        if current_max <= V:
            valid_prefix = True
        
        # Check suffix option
        valid_suffix = False
        # Query suffix tree [P-1, N-1] (assuming 0-based)
        max_pre_suf = prefix_tree.range_max(P-1, N)
        max_suf_suf = suffix_tree.range_max(P-1, N)
        current_max_suf = max(max_pre_suf, max_suf_suf)
        if current_max_suf <= V:
            valid_suffix = True
        
        new_dp = 0
        if valid_prefix:
            new_dp += dp
        if valid_suffix:
            new_dp += dp
        if new_dp != 0:
            new_dp %= MOD
        dp = new_dp
        
        # Update the trees if any option is valid
        if valid_prefix:
            # Update prefix_tree for [0, P-1] to max(current, V)
            for i in range(P):
                prefix_tree.update(i, V)
        if valid_suffix:
            # Update suffix_tree for [P-1, N-1] to max(current, V)
            for i in range(P-1, N):
                suffix_tree.update(i, V)
    
    print(dp % MOD)

if __name__ == '__main__':
    main()