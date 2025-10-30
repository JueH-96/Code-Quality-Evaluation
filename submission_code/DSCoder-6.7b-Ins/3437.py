class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Sort the power list in descending order
        power.sort(reverse=True)

        # Initialize the maximum total damage to 0
        max_total_damage = 0

        # Initialize the total damage to 0
        total_damage = 0

        # Initialize the used spells to an empty set
        used_spells = set()

        # Iterate over the sorted power list
        for i in range(len(power)):
            # If the spell has not been used before
            if power[i] not in used_spells:
                # Add the spell's damage to the total damage
                total_damage += power[i]

                # Add the spell to the used spells set
                used_spells.add(power[i])

                # If the spell can be cast with a damage of power[i] - 2
                if power[i] - 2 in used_spells:
                    # Remove the spell with damage power[i] - 2 from the total damage
                    total_damage -= power[i] - 2

                # If the spell can be cast with a damage of power[i] - 1
                if power[i] - 1 in used_spells:
                    # Remove the spell with damage power[i] - 1 from the total damage
                    total_damage -= power[i] - 1

                # If the spell can be cast with a damage of power[i] + 1
                if power[i] + 1 in used_spells:
                    # Remove the spell with damage power[i] + 1 from the total damage
                    total_damage -= power[i] + 1

                # If the spell can be cast with a damage of power[i] + 2
                if power[i] + 2 in used_spells:
                    # Remove the spell with damage power[i] + 2 from the total damage
                    total_damage -= power[i] + 2

            # Update the maximum total damage
            max_total_damage = max(max_total_damage, total_damage)

        # Return the maximum total damage
        return max_total_damage