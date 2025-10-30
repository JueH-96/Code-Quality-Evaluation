from typing import List
from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        m = n * (n + 1) // 2
        target = (m - 1) // 2
        
        max_unique = len(set(nums))
        low, high = 1, max_unique
        
        def count_at_most_k(k):
            count = 0
            left = 0
            freq = defaultdict(int)
            current_distinct = 0
            for right in range(n):
                num = nums[right]
                if freq[num] == 0:
                    current_distinct += 1
                freq[num] += 1
                while current_distinct > k:
                    left_num = nums[left]
                    freq[left_num] -= 1
                    if freq[left_num] == 0:
                        current_distinct -= 1
                    left += 1
                count += right - left + 1
            return count
        
        while low < high:
            mid = (low + high) // 2
            cnt = count_at_most_k(mid)
            if cnt >= target + 1:
                high = mid
            else:
                low = mid + 1
        return low