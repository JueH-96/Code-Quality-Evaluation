class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        prev_or_values = set()
        
        for num in nums:
            curr_or_values = {num}
            
            # For each previous OR value, compute OR with current number
            for prev_or in prev_or_values:
                curr_or_values.add(prev_or | num)
            
            # Update minimum difference for all current OR values
            for or_val in curr_or_values:
                min_diff = min(min_diff, abs(k - or_val))
            
            # Update prev_or_values for next iteration
            prev_or_values = curr_or_values
        
        return min_diff