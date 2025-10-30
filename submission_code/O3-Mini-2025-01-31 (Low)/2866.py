from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_length = 0
        for l in range(n):
            # The first condition: starting element must be even.
            if nums[l] % 2 != 0 or nums[l] > threshold:
                continue
            length = 1
            # Start checking subarray from index l onwards.
            for i in range(l, n - 1):
                current = nums[i]
                nxt = nums[i+1]
                if nxt > threshold:
                    break
                # Check parity alternating condition:
                if current % 2 == nxt % 2:
                    break
                length += 1
            max_length = max(max_length, length)
        return max_length