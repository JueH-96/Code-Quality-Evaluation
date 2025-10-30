class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        total_distinct = len(set(nums))
        n = len(nums)
        count = 0
        for i in range(n):
            subarray_elements = set()
            for j in range(i, n):
                subarray_elements.add(nums[j])
                if len(subarray_elements) == total_distinct:
                    count += 1
        return count