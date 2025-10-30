import bisect
from typing import List

class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        if x == 0:
            # Handle x=0 case: minimal consecutive difference in sorted array
            if len(nums) <= 1:
                return 0
            sorted_nums = sorted(nums)
            min_diff = float('inf')
            for i in range(len(sorted_nums) - 1):
                min_diff = min(min_diff, sorted_nums[i+1] - sorted_nums[i])
            return min_diff if min_diff != float('inf') else 0
        
        sorted_list = []
        min_diff = float('inf')
        n = len(nums)
        
        for j in range(n):
            # Insert nums[j - x] if j >= x
            if j >= x:
                val = nums[j - x]
                pos = bisect.bisect_left(sorted_list, val)
                sorted_list.insert(pos, val)
            
            # Query the sorted_list for closest elements to nums[j]
            if sorted_list:
                pos = bisect.bisect_left(sorted_list, nums[j])
                if pos < len(sorted_list):
                    min_diff = min(min_diff, abs(sorted_list[pos] - nums[j]))
                if pos > 0:
                    min_diff = min(min_diff, abs(sorted_list[pos - 1] - nums[j]))
        
        return min_diff if min_diff != float('inf') else 0