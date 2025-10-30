from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        # Coordinate compress nums
        vals = sorted(set(nums))
        comp = {v: i for i, v in enumerate(vals)}
        m = len(vals)
        
        # Fenwick Tree (BIT) for counts + find-by-order
        class BIT:
            def __init__(self, n):
                self.n = n
                self.tree = [0] * (n + 1)
            
            def update(self, i, delta):
                # i: 0-based index
                i += 1
                while i <= self.n:
                    self.tree[i] += delta
                    i += i & -i
            
            def query(self, i):
                # sum of [0..i], i may be -1
                if i < 0:
                    return 0
                i += 1
                s = 0
                while i > 0:
                    s += self.tree[i]
                    i -= i & -i
                return s
            
            def find_by_order(self, k):
                # find smallest p (1-based) with prefix_sum >= k
                # return 0-based index = p-1
                pos = 0
                # largest power of two <= n
                bitMask = 1 << (self.n.bit_length() - 1)
                while bitMask:
                    nxt = pos + bitMask
                    if nxt <= self.n and self.tree[nxt] < k:
                        k -= self.tree[nxt]
                        pos = nxt
                    bitMask >>= 1
                # pos is the largest index where prefix sum < original k
                # so the answer is pos+1 in 1-based, hence compress idx = pos
                return pos
        
        bit = BIT(m)
        ans = 10**18
        n = len(nums)
        
        # For each j from x..n-1, we insert nums[j-x] into BIT,
        # then query for nums[j].
        for j in range(x, n):
            # insert the element that now is at least x away
            v_ins = nums[j - x]
            c_ins = comp[v_ins]
            bit.update(c_ins, 1)
            
            v = nums[j]
            c = comp[v]
            
            # if there's already an equal value in BIT -> diff = 0
            cnt_eq = bit.query(c) - bit.query(c - 1)
            if cnt_eq > 0:
                return 0
            
            # predecessor: largest value < v
            cnt_less = bit.query(c - 1)
            if cnt_less > 0:
                pi = bit.find_by_order(cnt_less)
                diff = v - vals[pi]
                if diff < ans:
                    ans = diff
                    if ans == 0:
                        return 0
            
            # successor: smallest value > v
            total = bit.query(m - 1)
            cnt_le = bit.query(c)
            if cnt_le < total:
                # the (cnt_le+1)-th element in sorted order
                si = bit.find_by_order(cnt_le + 1)
                diff = vals[si] - v
                if diff < ans:
                    ans = diff
                    if ans == 0:
                        return 0
        
        return ans