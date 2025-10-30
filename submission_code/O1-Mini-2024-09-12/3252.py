class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                remaining = nums[:i] + nums[j+1:]
                if all(remaining[k] > remaining[k-1] for k in range(1, len(remaining))):
                    count += 1
        return count