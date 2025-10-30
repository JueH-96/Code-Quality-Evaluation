class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        distinct_elements_in_nums = set(nums)
        distinct_count = len(distinct_elements_in_nums)
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                distinct_elements_in_subarray = set(subarray)
                subarray_distinct_count = len(distinct_elements_in_subarray)
                if subarray_distinct_count == distinct_count:
                    count += 1
        return count