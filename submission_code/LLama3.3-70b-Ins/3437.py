from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Sort the power array in descending order
        power.sort(reverse=True)
        
        # Initialize a set to store the used powers
        used = set()
        
        # Initialize the total damage
        total_damage = 0
        
        # Iterate over the sorted power array
        for p in power:
            # Check if the current power and its adjacent powers are not used
            if p not in used and p-1 not in used and p-2 not in used and p+1 not in used and p+2 not in used:
                # Add the current power to the total damage
                total_damage += p
                # Add the current power and its adjacent powers to the used set
                used.add(p)
                used.add(p-1)
                used.add(p-2)
                used.add(p+1)
                used.add(p+2)
        
        # Return the maximum possible total damage
        return total_damage