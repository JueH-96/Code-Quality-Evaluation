class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        distinct_count = len(set(nums))
        count = 0
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                sub_array = nums[i:j+1]
                if len(set(sub_array)) == distinct_count:
                    count += 1
        return count