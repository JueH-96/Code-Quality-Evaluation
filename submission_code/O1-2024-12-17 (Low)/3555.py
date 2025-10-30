class Solution:
    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for _ in range(k):
            # Find the index of the first occurrence of the minimum value
            min_index = 0
            min_val = nums[0]
            for i in range(1, len(nums)):
                if nums[i] < min_val:
                    min_val = nums[i]
                    min_index = i
            
            # Replace the minimum value with min_val * multiplier
            nums[min_index] = nums[min_index] * multiplier
        
        return nums