from typing import List
from collections import Counter

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Count the occurrences of each power of 2 in nums
        power_count = Counter(nums)
        
        # Start with zero operations
        operations = 0
        
        # Iterate over powers of 2 from 1 to 2^30
        for power in range(31):
            current_power = 1 << power  # This is 2^power
            
            # If the current power of 2 is needed in the target
            if target & current_power:
                # If we have enough of this power in nums
                if power_count[current_power] > 0:
                    power_count[current_power] -= 1
                else:
                    # We need to perform operations to create this power
                    # Look for a higher power to split
                    found = False
                    for higher_power in range(power + 1, 31):
                        if power_count[1 << higher_power] > 0:
                            # We found a higher power, perform operations to split it
                            for split_power in range(higher_power, power, -1):
                                power_count[1 << split_power] -= 1
                                power_count[1 << (split_power - 1)] += 2
                                operations += 1
                            power_count[current_power] -= 1
                            found = True
                            break
                    if not found:
                        return -1
        
        return operations