from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        # Count how many k's we already have
        total_k = nums.count(k)
        # The best we can do is at least total_k (e.g. choose x=0 on a single-element segment)
        ans = total_k
        n = len(nums)
        # Only values in nums matter, and we skip v == k (x would be zero)
        uniq = set(nums)
        for v in uniq:
            if v == k:
                continue
            # For x = (k - v), we gain +1 for each v in the segment
            # and lose 1 for each original k in the segment (they become k + x != k).
            # All other numbers in the segment neither help nor hurt.
            curr = 0
            best = 0
            for num in nums:
                if num == v:
                    curr += 1
                elif num == k:
                    curr -= 1
                # otherwise delta = 0, no change
                # Kadane's reset
                if curr < 0:
                    curr = 0
                if curr > best:
                    best = curr
            # total after operation = original k's outside + converted v's = total_k - lost_k + best
            # but lost_k is counted in the Kadane loss, so adding best directly:
            ans = max(ans, total_k + best)
        return ans