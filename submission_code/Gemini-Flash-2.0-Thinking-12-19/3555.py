class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            min_val = min(nums)
            min_index = -1
            for i in range(len(nums)):
                if nums[i] == min_val:
                    min_index = i
                    break
            nums[min_index] = nums[min_index] * multiplier
        return nums