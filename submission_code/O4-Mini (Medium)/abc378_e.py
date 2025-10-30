import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # Binary Indexed Tree (Fenwick) for counts and sums
    class BIT:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (size + 1)
        def update(self, i, v):
            # i: 1-based index
            while i <= self.n:
                self.tree[i] += v
                i += i & -i
        def query(self, i):
            # sum from 1..i
            s = 0
            while i > 0:
                s += self.tree[i]
                i -= i & -i
            return s

    # We need two BITs over values 0..M-1 (we map v -> idx = v+1)
    bit_count = BIT(M + 1)
    bit_sum   = BIT(M + 1)

    # Initialize with prefix sum P[0] = 0
    # count of 0 is 1, sum is 0
    bit_count.update(1, 1)
    # bit_sum.update(1, 0)  # no need, it's zero

    prev_count = 1  # how many prefix sums we've inserted so far
    prev_sum = 0    # sum of those prefix-sum values

    ans = 0
    prefix = 0
    # iterate j = 1..n
    for x in A:
        prefix = (prefix + x) % M
        idx = prefix + 1  # 1-based index in BIT
        
        # c1 = count of previous P[i] <= prefix
        c1 = bit_count.query(idx)
        # s1 = sum of those P[i]
        s1 = bit_sum.query(idx)
        
        c2 = prev_count - c1
        s2 = prev_sum - s1
        
        # sum over y <= prefix of (prefix - y)
        sum1 = c1 * prefix - s1
        # sum over y > prefix of (prefix - y + M)
        sum2 = c2 * (prefix + M) - s2
        
        ans += (sum1 + sum2)
        
        # insert this prefix into BITs
        bit_count.update(idx, 1)
        bit_sum.update(idx, prefix)
        prev_count += 1
        prev_sum += prefix

    print(ans)

if __name__ == "__main__":
    main()