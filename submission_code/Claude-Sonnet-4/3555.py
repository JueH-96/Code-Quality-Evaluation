class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            # Find the minimum value
            min_val = min(nums)
            # Find the first index of the minimum value
            min_index = nums.index(min_val)
            # Replace with multiplied value
            nums[min_index] = min_val * multiplier
        
        return nums