class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        """
        Find the minimum possible value of the bitwise OR of the remaining elements
        after applying at most k operations.
        
        The approach is to check each bit from most significant to least significant.
        For each bit, we try to ensure it's 0 in the final OR. If we can achieve this
        with at most k operations, we keep it as 0. Otherwise, we set it to 1.
        """
        
        def check(mask, k):
            """
            Check if it's possible to ensure the final OR is at most 'mask'
            with at most k operations.
            """
            ops_needed = 0
            curr_and = -1  # All bits set initially
            
            for num in nums:
                curr_and &= num
                
                # If all bits outside mask are cleared, we reset the current AND chain
                if (curr_and | mask) == mask:
                    curr_and = -1
                else:
                    ops_needed += 1
            
            # Adjust for the last operation
            if curr_and != -1:
                ops_needed -= 1
            
            return ops_needed <= k
        
        result = 0
        
        # Check each bit from most significant to least significant
        for bit in range(29, -1, -1):
            # Try to keep the current bit as 0 in the final OR
            if not check(result, k):
                # If not possible, set the bit to 1 in the result
                result |= (1 << bit)
        
        return result