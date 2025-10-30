class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        max_num = max(nums)
        count = 0
        subarray_count = 0
        for num in nums:
            if num == max_num:
                count += 1
                subarray_count += count
            else:
                subarray_count += count
        return subarray_count