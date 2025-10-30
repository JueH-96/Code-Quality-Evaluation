from typing import List
from bisect import bisect_left

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        count = [0] * (n + 1)
        
        def check(k):
            last_seen = {}
            unique_count = 0
            for i in range(n):
                if i > k:
                    if last_seen[nums[i - k - 1]] == i - k - 1:
                        unique_count -= 1
                if nums[i] not in last_seen or last_seen[nums[i]] < i - k:
                    unique_count += 1
                last_seen[nums[i]] = i
                if unique_count <= k:
                    count[k] += 1
            return count[k] >= (n * (n + 1) // 2 + 1) // 2
        
        left, right = 0, n
        while left < right:
            mid = (left + right) // 2
            if check(mid):
                right = mid
            else:
                left = mid + 1
        return left