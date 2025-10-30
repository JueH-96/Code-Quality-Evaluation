class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        
        for i in range(len(nums)):
            # Set to store all possible OR values ending at current position
            current_ors = set()
            
            for j in range(i, len(nums)):
                # Update OR values by including nums[j]
                new_ors = {nums[j]}  # Start with single element
                
                for or_val in current_ors:
                    new_ors.add(or_val | nums[j])
                
                current_ors = new_ors
                
                # Find minimum difference among all current OR values
                for or_val in current_ors:
                    min_diff = min(min_diff, abs(k - or_val))
        
        return min_diff