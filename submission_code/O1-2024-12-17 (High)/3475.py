class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        # We'll use a difference array (diff) to track how flips 
        # propagate so we don't repeatedly flip the same elements.
        diff = [0] * (n+1)
        flips = 0     # Count how many flips (operations) we've made
        curr = 0      # Current cumulative flips in effect at position i
        
        for i in range(n):
            curr += diff[i]        # Update how many flips are affecting this index
            effective_val = nums[i] ^ (curr & 1)  # The bit value after flips
            
            # If the bit is 0 (and we want 1), we must flip at i..i+2
            if effective_val == 0:
                # If there's no space to flip 3 consecutive bits, it's impossible
                if i + 2 >= n:
                    return -1
                flips += 1
                curr += 1          # This start flip affects subsequent positions
                # Mark where the flip effect ends (i+3)
                if i + 3 < n:
                    diff[i+3] -= 1
        
        return flips