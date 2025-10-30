class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_or = 0
        for i in range(n):
            original_val = nums[i]
            for j in range(k + 1):
                nums[i] = original_val * (1 << j)
                current_or = 0
                for num in nums:
                    current_or |= num
                max_or = max(max_or, current_or)
                nums[i] = original_val
        return max_or