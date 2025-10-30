from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        ans = float('inf')
        # 'prev' holds all distinct OR results of subarrays ending at the previous index.
        prev = set()
        
        for num in nums:
            # Start a new set for this position including the current element itself.
            curr = {num}
            # Extend all previous subarrays ending at the last index by OR-ing with the current element.
            for val in prev:
                curr.add(val | num)
            
            # Update the answer with the absolute difference at the current step.
            for val in curr:
                ans = min(ans, abs(k - val))
            
            # Prepare for the next iteration.
            prev = curr
        
        return ans