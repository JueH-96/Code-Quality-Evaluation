from collections import Counter

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        unique_elements = len(set(nums))
        count = 0
        
        for i in range(n):
            subarray_counter = Counter()
            for j in range(i, n):
                subarray_counter[nums[j]] += 1
                if len(subarray_counter) == unique_elements:
                    count += 1
        
        return count