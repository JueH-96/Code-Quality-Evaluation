class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        changes = 0
        for i in range(n // 2):
            if nums[i] != nums[n - i - 1]:
                changes += 1
        return changes