class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        count = 0
        unique_nums = len(set(nums))
        for i in range(len(nums)):
            distinct = set()
            for j in range(i, len(nums)):
                distinct.add(nums[j])
                if len(distinct) == unique_nums:
                    count += 1
        return count