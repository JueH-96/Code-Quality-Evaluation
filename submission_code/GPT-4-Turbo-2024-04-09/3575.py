class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        from itertools import combinations
        
        def or_value(seq):
            result = 0
            for num in seq:
                result |= num
            return result
        
        n = len(nums)
        max_value = 0
        
        # Generate all combinations of 2*k elements from nums
        for comb in combinations(nums, 2 * k):
            # Split the combination into two halves
            first_half = comb[:k]
            second_half = comb[k:]
            
            # Calculate OR value for each half
            or_first_half = or_value(first_half)
            or_second_half = or_value(second_half)
            
            # Calculate XOR of the OR values of the two halves
            current_value = or_first_half ^ or_second_half
            
            # Update max_value if the current_value is greater
            max_value = max(max_value, current_value)
        
        return max_value