class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                remaining_nums = nums[:i] + nums[j+1:]
                is_strictly_increasing = True
                if len(remaining_nums) > 1:
                    for k in range(1, len(remaining_nums)):
                        if remaining_nums[k] <= remaining_nums[k-1]:
                            is_strictly_increasing = False
                            break
                if is_strictly_increasing:
                    count += 1
        return count