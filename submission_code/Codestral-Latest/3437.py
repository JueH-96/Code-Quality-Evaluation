class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from collections import defaultdict

        # Count the frequency of each damage value
        damage_count = defaultdict(int)
        for damage in power:
            damage_count[damage] += 1

        # Sort the unique damage values
        unique_damages = sorted(damage_count.keys())

        # Initialize variables to keep track of the maximum damage and the previous damage
        max_damage = 0
        prev_damage = float('-inf')

        for damage in unique_damages:
            if damage - 2 > prev_damage:
                # We can use all spells of this damage
                max_damage += damage * damage_count[damage]
                prev_damage = damage
            elif damage - 1 > prev_damage:
                # We can use all spells of this damage
                max_damage += damage * damage_count[damage]
                prev_damage = damage - 1
            elif damage > prev_damage:
                # We can use all spells of this damage
                max_damage += damage * damage_count[damage]
                prev_damage = damage

        return max_damage