class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        result = []
        
        for num in nums:
            # Find the rightmost bit that can be turned off
            # We need to find x such that x | (x + 1) = num
            
            # Key insight: x | (x + 1) sets all bits from the rightmost 0 in x to the end
            # So num should have a block of consecutive 1s at the end
            
            found = False
            
            # Try turning off each 1 bit from right to left
            for i in range(20):  # sufficient for the constraints
                if (num >> i) & 1:  # if bit i is set
                    # Try candidate where we turn off bit i
                    candidate = num ^ (1 << i)  # flip bit i
                    
                    if candidate | (candidate + 1) == num:
                        result.append(candidate)
                        found = True
                        break
            
            if not found:
                result.append(-1)
                
        return result