from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        prev = set()
        ans = float('inf')
        for num in nums:
            # Start new set of ORs ending at this position with just the current number
            new_prev = {num}
            # Update answer with the single-element subarray
            diff = abs(num - k)
            if diff < ans:
                ans = diff
                if ans == 0:
                    return 0
            # Extend all previous subarrays by OR-ing with current number
            for v in prev:
                val = v | num
                if val not in new_prev:
                    new_prev.add(val)
                    diff = abs(val - k)
                    if diff < ans:
                        ans = diff
                        if ans == 0:
                            return 0
            prev = new_prev
        return ans