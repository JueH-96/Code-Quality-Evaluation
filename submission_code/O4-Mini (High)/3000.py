from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        # If x == 0, we can pick the same element twice and difference is 0
        if x == 0:
            return 0
        
        n = len(nums)
        # Coordinate-compress the values in nums
        vals = sorted(set(nums))
        m = len(vals)
        comp = {v: i+1 for i, v in enumerate(vals)}  # 1-based indices for BIT
        
        # Fenwick Tree / BIT for frequencies
        class BIT:
            def __init__(self, size: int):
                self.n = size
                self.tree = [0] * (size + 1)
            
            def update(self, i: int, delta: int):
                while i <= self.n:
                    self.tree[i] += delta
                    i += i & -i
            
            def query(self, i: int) -> int:
                """Sum of frequencies from 1..i"""
                s = 0
                while i > 0:
                    s += self.tree[i]
                    i -= i & -i
                return s
            
            def find_kth(self, k: int) -> int:
                """
                Find the smallest index i such that
                prefix-sum(1..i) >= k.
                Assumes 1 <= k <= total-frequency.
                """
                idx = 0
                # largest power of two <= n
                bit_mask = 1 << (self.n.bit_length() - 1)
                while bit_mask > 0:
                    nxt = idx + bit_mask
                    if nxt <= self.n and self.tree[nxt] < k:
                        k -= self.tree[nxt]
                        idx = nxt
                    bit_mask >>= 1
                return idx + 1
        
        bit = BIT(m)
        total_inserted = 0
        ans = float('inf')
        
        # For each j >= x, we insert nums[j-x] into our BIT,
        # then query the nearest neighbor of nums[j].
        for j in range(x, n):
            # Insert the element at index j-x into the BIT
            id_add = comp[nums[j - x]]
            bit.update(id_add, 1)
            total_inserted += 1
            
            # Now query for nums[j]
            v = nums[j]
            id_v = comp[v]
            
            # 1) predecessor: largest value <= v
            cnt_le = bit.query(id_v)
            if cnt_le > 0:
                idx_pred = bit.find_kth(cnt_le)
                v_pred = vals[idx_pred - 1]
                diff = abs(v - v_pred)
                if diff < ans:
                    ans = diff
                    if ans == 0:
                        return 0
            
            # 2) successor: smallest value >= v
            cnt_less = bit.query(id_v - 1)
            if cnt_less < total_inserted:
                idx_succ = bit.find_kth(cnt_less + 1)
                v_succ = vals[idx_succ - 1]
                diff = abs(v - v_succ)
                if diff < ans:
                    ans = diff
                    if ans == 0:
                        return 0
        
        return ans