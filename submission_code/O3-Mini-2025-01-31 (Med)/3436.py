from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        # Initialize the answer to a large number.
        ans = float('inf')
        # dp will hold all distinct OR values for subarrays ending at the previous index.
        dp = set()
        
        for num in nums:
            # new_dp will hold all distinct OR values for subarrays ending at the current index.
            new_dp = {num}
            for prev in dp:
                new_dp.add(prev | num)
            # Update the answer by checking each OR value computed.
            for val in new_dp:
                ans = min(ans, abs(k - val))
            dp = new_dp
        
        return ans