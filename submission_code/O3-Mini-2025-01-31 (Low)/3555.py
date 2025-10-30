class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            # Find the minimum value and its first index in nums
            min_val = min(nums)
            index = nums.index(min_val)
            # Replace the element with its multiplied value
            nums[index] = nums[index] * multiplier
        return nums