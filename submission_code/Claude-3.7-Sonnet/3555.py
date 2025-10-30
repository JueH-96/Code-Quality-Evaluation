class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            # Find the minimum value and its index in a single pass
            min_val = nums[0]
            min_idx = 0
            
            for i in range(1, len(nums)):
                if nums[i] < min_val:
                    min_val = nums[i]
                    min_idx = i
            
            # Replace the minimum value with its product with the multiplier
            nums[min_idx] = min_val * multiplier
        
        return nums