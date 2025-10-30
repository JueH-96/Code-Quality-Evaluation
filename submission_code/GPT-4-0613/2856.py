class Solution:
    def countCompleteSubarrays(self, nums):
        n = len(nums)
        distinct_count = len(set(nums))
        count = 0
        for i in range(n):
            distinct_elements = set()
            for j in range(i, n):
                distinct_elements.add(nums[j])
                if len(distinct_elements) == distinct_count:
                    count += 1
        return count