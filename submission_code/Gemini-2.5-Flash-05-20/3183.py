from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        k_or_result = 0
        
        # Iterate through possible bit positions.
        # Since nums[i] < 2^31, the numbers can be represented with up to 31 bits.
        # The highest possible bit index is 30 (for 2^30).
        # So we check bits from 0 up to 30.
        for i in range(31): 
            count_set_bits = 0
            
            # For each bit position 'i', count how many numbers in 'nums'
            # have this bit set.
            for num in nums:
                # Check if the i-th bit of 'num' is set.
                # (num >> i) shifts the num right by i positions, bringing the i-th bit to the 0th position.
                # Then, & 1 checks if this 0th bit is 1.
                if (num >> i) & 1:
                    count_set_bits += 1
            
            # If the count of numbers with the i-th bit set is at least 'k',
            # then the i-th bit should be set in the K-or result.
            if count_set_bits >= k:
                # Set the i-th bit in k_or_result.
                # (1 << i) creates a number with only the i-th bit set.
                # Using bitwise OR (|) ensures that this bit is set without affecting others.
                k_or_result |= (1 << i)
                
        return k_or_result