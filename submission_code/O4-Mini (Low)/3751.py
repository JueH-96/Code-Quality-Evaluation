from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # total original count of k
        total_k = nums.count(k)
        
        max_gain = 0
        
        # For each candidate value v (the value we will raise to k in a segment)
        # v == k corresponds to x = 0, gain = 0, so we can skip it
        for v in range(1, 51):
            if v == k:
                continue
            # We define an array delta where:
            # delta[i] = +1 if nums[i] == v (we gain one k)
            # delta[i] = -1 if nums[i] == k (we lose one k inside segment)
            # otherwise delta[i] = 0
            curr = 0
            best = 0
            for a in nums:
                if a == v:
                    curr += 1
                elif a == k:
                    curr -= 1
                # otherwise no change to curr
                # If curr goes negative, restart segment
                if curr < 0:
                    curr = 0
                # track best subarray sum for this v
                if curr > best:
                    best = curr
            # best is the max gain (B - A) for transforming v->k
            if best > max_gain:
                max_gain = best
        
        # The answer is original k's plus best possible gain
        return total_k + max_gain