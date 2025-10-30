class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            distinct = set()
            for j in range(i, n):
                distinct.add(nums[j])
                if len(distinct) == len(nums):
                    count += 1
                elif len(distinct) > len(nums):
                    break
        return count