from typing import List
from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        m = n * (n + 1) // 2
        k = (m - 1) // 2  # 0-based index for the median
        
        def count_at_most(x):
            freq = defaultdict(int)
            left = 0
            distinct = 0
            count = 0
            for right in range(n):
                num = nums[right]
                if freq[num] == 0:
                    distinct += 1
                freq[num] += 1
                
                # Shrink the window until distinct <= x
                while distinct > x:
                    left_num = nums[left]
                    freq[left_num] -= 1
                    if freq[left_num] == 0:
                        distinct -= 1
                    left += 1
                
                # Add the number of valid subarrays ending at right
                count += right - left + 1
            return count
        
        low = 1
        high = len(set(nums))
        ans = high
        
        while low <= high:
            mid = (low + high) // 2
            c = count_at_most(mid)
            if c >= k + 1:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return ans