from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        def count_at_most(max_distinct):
            left = 0
            freq = {}
            distinct_count = 0
            subarray_count = 0
            for right in range(len(nums)):
                if nums[right] not in freq:
                    freq[nums[right]] = 0
                if freq[nums[right]] == 0:
                    distinct_count += 1
                freq[nums[right]] += 1
                while distinct_count > max_distinct and left <= right:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        distinct_count -= 1
                    left += 1
                subarray_count += (right - left + 1)
            return subarray_count
        
        n = len(nums)
        m = n * (n + 1) // 2
        k = (m - 1) // 2
        target = k + 1
        low = 1
        high = n
        while low <= high:
            mid = (low + high) // 2
            if count_at_most(mid) >= target:
                high = mid - 1
            else:
                low = mid + 1
        return low