from typing import List
from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        freq = defaultdict(int)
        distinct = 0
        window_sum = 0
        result = 0
        
        for i, num in enumerate(nums):
            # add new element to window
            window_sum += num
            if freq[num] == 0:
                distinct += 1
            freq[num] += 1
            
            # if window is now too big, remove the oldest element
            if i >= k:
                old = nums[i - k]
                window_sum -= old
                freq[old] -= 1
                if freq[old] == 0:
                    del freq[old]
                    distinct -= 1
            
            # once we have a full window of size k, check distinct count
            if i >= k - 1 and distinct >= m:
                result = max(result, window_sum)
        
        return result