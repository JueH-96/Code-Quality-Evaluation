class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        def is_strictly_increasing(arr):
            for i in range(1, len(arr)):
                if arr[i] <= arr[i - 1]:
                    return False
            return True

        n = len(nums)
        count = 0
        for l in range(n):
            for r in range(l, n):
                new_nums = nums[:l] + nums[r + 1:]
                if is_strictly_increasing(new_nums):
                    count += 1
        return count