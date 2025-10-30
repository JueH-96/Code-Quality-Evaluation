from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        S = sum(nums)
        # M is the remainder we need from a partial cycle
        M = target % S
        
        # If we need exactly whole cycles (no partial), answer is full cycles * n
        if M == 0:
            # target >= S because target>=1 and S>=1, if target < S then M != 0
            return (target // S) * n
        
        # Otherwise we need a partial sum equal to M in a single (possibly wrapping) cycle
        # Use sliding window on a doubled array of length 2n, window length <= n
        best = n + 1
        curr_sum = 0
        left = 0
        # scan right end over 0..2n-1
        for right in range(2 * n):
            curr_sum += nums[right % n]
            # shrink window while sum > M or window too long
            while left <= right and (curr_sum > M or (right - left + 1) > n):
                curr_sum -= nums[left % n]
                left += 1
            # check match
            if curr_sum == M:
                length = right - left + 1
                if length < best:
                    best = length
        
        if best == n + 1:
            return -1
        
        # number of full cycles we can take
        k = target // S
        return k * n + best