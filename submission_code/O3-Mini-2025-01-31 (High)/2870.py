from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = -1
        curr_len = 0  # current alternating subarray length; 0 means not tracking a valid one
        base = None   # the first element of the current subarray
        
        for i in range(1, n):
            # If not currently in a valid segment, check if we can start one with nums[i-1], nums[i]
            if curr_len == 0:
                if nums[i] == nums[i-1] + 1:
                    curr_len = 2
                    base = nums[i-1]
                    max_len = max(max_len, curr_len)
                # Otherwise, nothing to do: curr_len remains 0
            else:
                # We are in an alternating segment started at some index j, with base = nums[j]
                # The next element (i) should be:
                # if curr_len is even: it is the (curr_len)th element (0-indexed position curr_len), should equal base.
                # if curr_len is odd: it should equal base+1.
                expected = base if curr_len % 2 == 0 else base + 1
                if nums[i] == expected:
                    curr_len += 1
                    max_len = max(max_len, curr_len)
                else:
                    # The current alternating subarray is broken.
                    # But that break might be the start of a new alternating subarray if nums[i-1] and nums[i] form a valid pair.
                    if nums[i] == nums[i-1] + 1:
                        curr_len = 2
                        base = nums[i-1]
                        max_len = max(max_len, curr_len)
                    else:
                        curr_len = 0
                        base = None
                        
        return max_len