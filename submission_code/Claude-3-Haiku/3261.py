class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        while k > 0:
            min_idx = 0
            for i in range(len(nums) - 1):
                if nums[i] & nums[i+1] < nums[min_idx] & nums[min_idx+1]:
                    min_idx = i
            nums[min_idx] = nums[min_idx] & nums[min_idx+1]
            nums.pop(min_idx+1)
            k -= 1
        return min(nums) | max(nums)