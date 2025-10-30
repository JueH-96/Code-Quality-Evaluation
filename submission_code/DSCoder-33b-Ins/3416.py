class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        total_diff = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                diff = 0
                for k in range(len(str(nums[i]))):
                    diff += abs(int(str(nums[i])[k]) - int(str(nums[j])[k]))
                total_diff += diff
        return total_diff