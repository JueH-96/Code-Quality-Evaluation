class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        
        # Initialize the minimum difference to a very large value.
        min_diff = float('inf')
        
        # This set stores the bitwise ORs of all subarrays ending at the previous position.
        # The size of this set is bounded by the number of bits in the numbers (e.g., ~30).
        prev_ors = set()

        for num in nums:
            # Calculate the ORs of subarrays ending at the current position.
            # A subarray ending here is either just `[num]` or a previous one extended with `num`.
            current_ors = {num}
            for or_val in prev_ors:
                current_ors.add(or_val | num)
            
            # Update the minimum difference with the new OR values.
            for or_val in current_ors:
                min_diff = min(min_diff, abs(k - or_val))
            
            # If a difference of 0 is found, it's the minimum possible, so we can return immediately.
            if min_diff == 0:
                return 0
            
            # The current set of ORs becomes the "previous" set for the next iteration.
            prev_ors = current_ors
            
        return min_diff