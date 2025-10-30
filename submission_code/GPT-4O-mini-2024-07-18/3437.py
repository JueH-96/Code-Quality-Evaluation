from typing import List
from collections import Counter

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        damage_count = Counter(power)
        max_damage = 0
        
        # We will iterate through the unique spell damages in sorted order
        for damage in sorted(damage_count.keys()):
            # If we can cast this spell, we add its total damage
            max_damage += damage * damage_count[damage]
            # Remove the counts of the spells that cannot be cast
            damage_count[damage - 2] = 0
            damage_count[damage - 1] = 0
            damage_count[damage + 1] = 0
            damage_count[damage + 2] = 0
        
        return max_damage