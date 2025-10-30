class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            min_val = float('inf')
            min_index = -1
            for i in range(len(nums)):
                if nums[i] < min_val:
                    min_val = nums[i]
                    min_index = i
            nums[min_index] *= multiplier
        return nums