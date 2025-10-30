class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        or_values = set()
        
        for num in nums:
            # Create new set of OR values for subarrays ending at current position
            new_or_values = {num}  # Start new subarray with just current element
            
            # Extend all previous subarrays to include current element
            for or_val in or_values:
                new_or_values.add(or_val | num)
            
            # Update minimum difference
            for or_val in new_or_values:
                min_diff = min(min_diff, abs(k - or_val))
            
            or_values = new_or_values
        
        return min_diff