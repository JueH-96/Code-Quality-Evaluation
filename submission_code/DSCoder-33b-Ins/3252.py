class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i + 1, n + 1):
                subarray = nums[:i] + nums[j:]
                if all(subarray[k] < subarray[k + 1] for k in range(len(subarray) - 1)):
                    count += 1
        return count