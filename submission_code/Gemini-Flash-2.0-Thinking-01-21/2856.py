class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        distinct_elements_nums = set(nums)
        distinct_count_nums = len(distinct_elements_nums)
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                distinct_elements_subarray = set(subarray)
                distinct_count_subarray = len(distinct_elements_subarray)
                if distinct_count_subarray == distinct_count_nums:
                    count += 1
        return count