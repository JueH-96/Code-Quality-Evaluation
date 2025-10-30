def main():
    import sys,sys
    data = sys.stdin.buffer.read().split()
    if not data: 
        return
    it = iter(data)
    try:
        n = int(next(it))
        m = int(next(it))
    except StopIteration:
        return

    # Compute cycle length r.
    r = 1
    while r < m:
        r *= 2

    # Read sequences (each as a bytes object).
    sequences = []
    for _ in range(n):
        seq = bytes( int(next(it)) for _ in range(m) )
        sequences.append(seq)

    # diff function: difference operator (inverse of T)
    def diff(seq):
        l = len(seq)
        if l <= 1:
            return seq
        res = bytearray(l)
        res[0] = seq[0]
        prev = seq[0]
        for i in range(1, l):
            cur = seq[i]
            res[i] = cur ^ prev
            prev = cur
        return bytes(res)

    # For a given sequence, compute the canonical representative (the lexicographically smallest element in its orbit)
    # and the unique x in [0,r) such that diff^x(seq)==canonical.
    def canonical_log(seq):
        best = seq
        best_off = 0
        cur = seq
        for x in range(1, r):
            cur = diff(cur)
            if cur < best:
                best = cur
                best_off = x
        cur = seq
        x = 0
        while x < r:
            if cur == best:
                return best, x
            cur = diff(cur)
            x += 1
        return best, 0

    # Group sequences by their orbit (canonical rep) and record (input_index, discrete_log)
    groups = {}
    for idx, seq in enumerate(sequences):
        can, logval = canonical_log(seq)
        groups.setdefault(can, []).append((idx, logval))
        
    mod = 998244353
    ans = 0

    # We now sum contributions inside each group.
    class BIT:
        __slots__ = ('n', 'tree')
        def __init__(self, n):
            self.n = n
            self.tree = [0]*(n+1)
        def update(self, i, delta):
            i += 1
            while i <= self.n:
                self.tree[i] += delta
                i += i & -i
        def query(self, i):
            s = 0
            i += 1
            while i:
                s += self.tree[i]
                i -= i & -i
            return s

    # Process groups.
    for group in groups.values():
        group.sort(key=lambda x: x[0])
        if len(group) <= 1:
            continue
        bitCount = BIT(r)
        bitSum = BIT(r)
        totalCnt = 0
        totalSum = 0
        for _, exp_val in group:
            cnt_le = bitCount.query(exp_val)
            sum_le = bitSum.query(exp_val)
            cnt_all = totalCnt
            sum_all = totalSum
            cnt_gt = cnt_all - cnt_le
            sum_gt = sum_all - sum_le
            contrib = (cnt_le * exp_val - sum_le) + (cnt_gt * (exp_val + r) - sum_gt)
            ans = (ans + contrib) % mod
            bitCount.update(exp_val, 1)
            bitSum.update(exp_val, exp_val)
            totalCnt += 1
            totalSum += exp_val
    sys.stdout.write(str(ans % mod))
    
if __name__ == '__main__':
    main()