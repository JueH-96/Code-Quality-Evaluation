from collections import Counter
from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        res = 0
        
        # Iterate r from 2 to n-3 (so that p < q < r and s > r with gaps >=1)
        for r in range(2, n-1):
            # Build suffix_count for s in [r+2..n-1]
            suffix_count = Counter()
            for s in range(r+2, n):
                suffix_count[nums[s]] += 1
            
            # left_count will track counts of nums[p]*nums[r] for p in [0..q-2]
            left_count = Counter()
            # We will move q from 1..r-1 but need r-q>1 => q <= r-2
            # We add p = q-1 into left_count before using q
            for q in range(1, r-1):
                # include p = q-1 into left_count
                p = q-1
                prod_pr = nums[p] * nums[r]
                left_count[prod_pr] += 1
                
                # now q is valid; for each product L = nums[p]*nums[r],
                # we want nums[q] * nums[s] == L => nums[s] == L // nums[q], L % nums[q]==0
                vq = nums[q]
                # iterate over current left_count items
                for L, cnt_p in left_count.items():
                    if L % vq == 0:
                        need = L // vq
                        cnt_s = suffix_count.get(need, 0)
                        if cnt_s:
                            res += cnt_p * cnt_s
                
                # Before moving q->q+1, we will lose no info in suffix_count
                # because s and q are independent here
                # (we only use suffix_count which is static for this r)
        
        return res