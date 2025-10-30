from typing import List

class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        """
        Calculates the K-or of a list of integers.

        The K-or is a non-negative integer where the i-th bit is set 
        if and only if there are at least k elements in nums where bit i is set.
        """
        result = 0
        
        # Since the numbers are non-negative and less than 2^31,
        # we only need to consider bits from 0 to 30.
        for i in range(31):
            # For each bit position, count how many numbers have this bit set.
            count = 0
            for num in nums:
                # Check if the i-th bit is set in num.
                # A common way is to right-shift num by i and check the last bit.
                if (num >> i) & 1:
                    count += 1
            
            # If the count meets the threshold k, set the i-th bit in our result.
            if count >= k:
                # To set the i-th bit, we can use bitwise OR with (1 << i).
                result |= (1 << i)
                
        return result