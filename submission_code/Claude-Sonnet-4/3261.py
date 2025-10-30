class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        result = 0
        
        # Check each bit position from high to low (bit 29 to bit 0)
        for bit in range(29, -1, -1):
            # Try to make this bit 0 in the final result
            # Create a mask that includes all bits we want to keep as 0
            mask = result | ((1 << bit) - 1)
            
            # Count operations needed to make current bit 0
            operations = 0
            i = 0
            
            while i < n:
                if nums[i] & (1 << bit):  # If current number has this bit set
                    # Try to eliminate this bit by AND operations
                    current = nums[i]
                    j = i
                    
                    # Keep doing AND operations while we still have the bit set
                    # and we haven't reached the end
                    while j < n - 1 and (current & (1 << bit)):
                        current &= nums[j + 1]
                        operations += 1
                        j += 1
                    
                    # If we couldn't eliminate the bit, we can't make this bit 0
                    if current & (1 << bit):
                        operations = k + 1  # Mark as impossible
                        break
                    
                    i = j + 1
                else:
                    i += 1
            
            # If we can eliminate this bit with at most k operations
            if operations <= k:
                # We can make this bit 0, so don't add it to result
                k -= operations
            else:
                # We cannot make this bit 0, so it must be 1 in final result
                result |= (1 << bit)
        
        return result