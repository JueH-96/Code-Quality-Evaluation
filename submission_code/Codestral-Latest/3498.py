class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        changes = [0] * (k + 1)

        for i in range(n // 2):
            for x in range(k + 1):
                changes[x] = min(changes[x] + (nums[i] != x) + (nums[n - i - 1] != x),
                                 changes[x] + (nums[i] != x) + (nums[n - i - 1] != k - x),
                                 changes[x] + (nums[i] != k - x) + (nums[n - i - 1] != x),
                                 changes[x] + (nums[i] != k - x) + (nums[n - i - 1] != k - x))

        return min(changes)