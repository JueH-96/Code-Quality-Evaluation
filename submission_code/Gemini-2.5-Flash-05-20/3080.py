import math
from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)

        # Step 1: Calculate the bitwise AND of all elements in the entire array.
        # This determines the absolute minimum possible sum of scores.
        # Initialize overall_and with the first element.
        # If nums is empty, this would raise an IndexError, but constraints say 1 <= nums.length.
        overall_and = nums[0]
        for i in range(1, n):
            overall_and &= nums[i]
        
        # Step 2: If overall_and > 0, it means that at least one bit is set in every number
        # in the entire array. Therefore, any subarray, no matter how small, will also have
        # that bit set in its score. This implies that the score of any subarray s_j will be
        # s_j >= overall_and.
        # The sum of scores will be sum(s_j) >= m * overall_and, where m is the number of subarrays.
        # To minimize this sum (which is overall_and), we must have m=1.
        # This means the only way to achieve the minimum total score is to take the entire array
        # as a single subarray.
        if overall_and > 0:
            return 1
        
        # Step 3: If overall_and == 0, the minimum possible sum of scores is 0.
        # To achieve a total sum of 0, each individual subarray's score must be 0.
        # We want to maximize the number of such subarrays.
        # We use a greedy approach:
        # Iterate through the array, forming subarrays. Whenever the bitwise AND of the
        # current subarray becomes 0, we've found a valid subarray. We increment the count
        # and start a new subarray from the next element. This strategy maximizes the number
        # of subarrays by making each valid subarray as short as possible.

        count = 0
        # Initialize current_and with a value that has all possible bits set
        # relevant for numbers up to 10^6.
        # 10^6 is less than 2^20 (1048576), so 20 bits are sufficient.
        # (1 << 20) - 1 creates a bitmask with bits 0 through 19 all set.
        current_and_segment = (1 << 20) - 1 
        
        for num in nums:
            current_and_segment &= num
            # If the current segment's AND product becomes 0,
            # we've found a subarray with score 0.
            if current_and_segment == 0:
                count += 1
                # Reset current_and_segment to start a new subarray from the next element.
                current_and_segment = (1 << 20) - 1
                
        return count