class Solution:
    def incremovableSubarrayCount(self, nums):
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i, n):
                left = nums[:i]
                right = nums[j+1:]
                if left + right == sorted(set(left + right)):
                    count += 1
        return count