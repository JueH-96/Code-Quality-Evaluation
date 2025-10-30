from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # First, compute the overall bitwise AND of the entire array.
        overall = nums[0]
        for num in nums[1:]:
            overall &= num
        
        # If the overall AND is greater than 0, then no matter how we split,
        # each subarray’s AND is at least overall, so if we split into more than one,
        # the total sum would be > overall. Hence the only valid split is the whole array.
        if overall > 0:
            return 1
        
        # If overall == 0, our goal is to partition the array into as many subarrays as possible
        # such that each subarray’s bitwise AND equals 0.
        # The greedy strategy is to iterate over nums and maintain a running AND.
        # As soon as the running AND becomes 0, we can “cut” and start a new subarray.
        # We use "running = -1" as an identity value (since -1 has all bits set)
        # and for any non-negative integer x, (-1 & x) equals x.
        count = 0
        running = -1
        for num in nums:
            running &= num
            if running == 0:
                count += 1
                # Reset running to -1 so that the next number starts a fresh subarray.
                running = -1
        # If no valid cut was found (which won't happen if overall == 0 since the whole array AND is 0),
        # at worst the whole array is one valid subarray.
        return count if count > 0 else 1