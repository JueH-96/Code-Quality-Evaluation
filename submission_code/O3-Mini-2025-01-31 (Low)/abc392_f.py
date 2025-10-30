def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    P = [0]*(n+1)
    for i in range(1, n+1):
        P[i] = int(data[i])
    
    # We use a Binary Indexed Tree (Fenwick Tree) to efficiently
    # simulate placing numbers into positions. We have n positions,
    # all initially free (represented by a 1). In the reversed process,
    # for i = n downto 1, we want to choose the P[i]-th free position,
    # fill it with number i and mark it as occupied (update BIT by -1).
    
    class BIT:
        def __init__(self, n):
            self.n = n
            self.tree = [0]*(n+1)
        def update(self, i, delta):
            while i <= self.n:
                self.tree[i] += delta
                i += i & -i
        def query(self, i):
            s = 0
            while i:
                s += self.tree[i]
                i -= i & -i
            return s
        # find smallest index such that prefix sum >= k
        def findKth(self, k):
            idx = 0
            bitMask = 1 << (self.n.bit_length())  # highest power of 2 >= n
            while bitMask:
                tIdx = idx + bitMask
                if tIdx <= self.n and self.tree[tIdx] < k:
                    k -= self.tree[tIdx]
                    idx = tIdx
                bitMask //= 2
            return idx + 1

    # initialize BIT with all positions free (1)
    bit = BIT(n)
    for i in range(1, n+1):
        bit.update(i, 1)
    
    ans = [0]*(n+1)  # 1-indexed
    # process in reverse order
    for i in range(n, 0, -1):
        pos = bit.findKth(P[i])
        ans[pos] = i
        bit.update(pos, -1)
    
    # output the final result, positions 1 to n in order.
    sys.stdout.write(" ".join(map(str, ans[1:])))
    
if __name__ == '__main__':
    main()