class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        min_diff = float('inf')
        prev_ors = set()  # Set of OR values for subarrays ending at previous position
        
        for num in nums:
            # Start with current element as single-element subarray
            curr_ors = {num}
            
            # Extend all previous OR values with current element
            for or_val in prev_ors:
                curr_ors.add(or_val | num)
            
            # Update minimum difference
            for or_val in curr_ors:
                min_diff = min(min_diff, abs(k - or_val))
                if min_diff == 0:
                    return 0
            
            prev_ors = curr_ors
        
        return min_diff