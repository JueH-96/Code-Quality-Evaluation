class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                if nums[j] > nums[i]:
                    break
                elif nums[j] == nums[i]:
                    count += 1
        return count