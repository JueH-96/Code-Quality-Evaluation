from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Count the frequency of each power of 2 in nums
        freq = Counter(nums)
        
        # Initialize the number of operations to 0
        operations = 0
        
        # Start from the largest power of 2 that is less than or equal to target
        power = 1 << (target.bit_length() - 1)
        
        while target > 0 and power > 0:
            # If the current power of 2 can be used, use as much as possible
            if freq[power] > 0:
                times = min(target // power, freq[power])
                target -= times * power
                freq[power] -= times
            
            # If the current power of 2 is greater than 1 and we still need more to reach the target
            if power > 1 and target >= power:
                # Split the current power of 2 into two halves
                freq[power // 2] += 2 * freq[power]
                operations += freq[power]
                freq[power] = 0
            
            # Move to the next smaller power of 2
            power //= 2
        
        # If we have reached the target, return the number of operations
        if target == 0:
            return operations
        # If it is impossible to reach the target, return -1
        return -1

# Example usage:
# sol = Solution()
# print(sol.minOperations([1,2,8], 7)) # Output: 1
# print(sol.minOperations([1,32,1,2], 12)) # Output: 2
# print(sol.minOperations([1,32,1], 35)) # Output: -1