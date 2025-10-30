from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Bitwise AND of the whole array
        total_and = nums[0]
        for x in nums[1:]:
            total_and &= x
        
        # If the AND of the whole array is positive, 
        # the minimal possible sum is that value and it is obtained
        # only by taking the whole array as a single sub-array.
        if total_and != 0:
            return 1
        
        # We now know the minimal sum is 0.
        # We want as many sub-arrays as possible,
        # each having bitwise AND equal to 0.
        ALL_ONES = (1 << 30) - 1          # 30 ones – more than enough for nums[i] ≤ 10^6
        cur_and  = ALL_ONES
        answer   = 0
        
        for v in nums:
            cur_and &= v
            # Whenever the running AND becomes 0 we can close a sub-array here
            # (earliest possible) to maximise the count.
            if cur_and == 0:
                answer += 1
                cur_and = ALL_ONES        # start a new segment
        
        return answer