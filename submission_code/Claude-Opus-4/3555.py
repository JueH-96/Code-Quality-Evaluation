class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            # Find the minimum value
            min_val = min(nums)
            
            # Find the index of the first occurrence of the minimum value
            min_idx = nums.index(min_val)
            
            # Multiply the value at that index by the multiplier
            nums[min_idx] *= multiplier
        
        return nums