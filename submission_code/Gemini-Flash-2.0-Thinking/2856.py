class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        distinct_elements_whole_array = len(set(nums))
        count = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                distinct_elements_subarray = len(set(subarray))
                if distinct_elements_subarray == distinct_elements_whole_array:
                    count += 1
        return count