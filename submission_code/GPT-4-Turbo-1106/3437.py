from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        damage = 0
        i = 0
        while i < len(power):
            damage += power[i]
            # Skip all spells within the forbidden range
            while i + 1 < len(power) and power[i + 1] <= power[i] + 2:
                i += 1
            i += 1
        return damage

# Example usage:
# sol = Solution()
# print(sol.maximumTotalDamage([1,1,3,4]))  # Output: 6
# print(sol.maximumTotalDamage([7,1,6,6]))  # Output: 13