class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        from itertools import combinations
        
        max_value = 0
        
        # Generate all combinations of size 2 * k
        for seq in combinations(nums, 2 * k):
            # Split the sequence into two halves
            first_half = seq[:k]
            second_half = seq[k:]
            
            # Calculate the OR for both halves
            first_or = 0
            for num in first_half:
                first_or |= num
            
            second_or = 0
            for num in second_half:
                second_or |= num
            
            # Calculate the value of the sequence
            value = first_or ^ second_or
            
            # Update the maximum value found
            max_value = max(max_value, value)
        
        return max_value