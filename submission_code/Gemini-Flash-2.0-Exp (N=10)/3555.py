class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            min_val = float('inf')
            min_index = -1
            for i, num in enumerate(nums):
                if num < min_val:
                    min_val = num
                    min_index = i
            nums[min_index] *= multiplier
        return nums