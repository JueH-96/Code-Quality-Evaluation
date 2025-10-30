class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        """
        We can repeatedly 'attack' an unmarked enemy (if we have enough energy) to gain points,
        and optionally 'absorb' an unmarked enemy (if we have at least 1 point) to increase our energy.
        
        Key observations:
          • An enemy remains unmarked after any number of attacks on it. It becomes marked only when absorbed.
          • Attacking an enemy with energy cost E reduces our currentEnergy by E and increases our points by 1.
          • Absorbing an enemy with energy cost E requires us to have ≥ 1 point (but does NOT consume a point)
            and increases our currentEnergy by E, then marks that enemy so it cannot be used again.
          • We want the maximum total points achievable.

        Strategy (greedy):
          1) Sort enemies' energies in non-decreasing order.
          2) Maintain two pointers: left at the smallest enemy, right at the largest.
          3) Repeatedly:
             - "Farm" the smallest unmarked enemy (enemyEnergies[left]) as many times as possible
               by attacking it. Each attack costs E[left] energy and yields +1 point.
             - If we can no longer attack the smallest enemy (i.e., currentEnergy < E[left]), 
               and we have at least 1 point and left < right, we absorb the largest enemy 
               (enemyEnergies[right]) to increase energy, then mark that largest enemy (right -= 1).
             - If neither is possible, we stop.
          4) Since absorbing does not spend (reduce) points, our total points will never decrease.
             We simply return the total points at the end.
        
        This approach covers:
          • Multiple attacks on a single small enemy (maximizing the number of attacks for our energy).
          • Occasional absorption of a large enemy if needed to replenish energy and continue attacking.
        """

        enemyEnergies.sort()
        left, right = 0, len(enemyEnergies) - 1
        points = 0

        while left <= right:
            # Attack repeatedly on the smallest unmarked enemy if possible
            cost = enemyEnergies[left]
            can_attack_times = currentEnergy // cost  # number of attacks we can afford on this enemy

            if can_attack_times > 0:
                # Perform all those attacks
                points += can_attack_times
                currentEnergy -= can_attack_times * cost
            else:
                # If we cannot attack, try absorbing the largest if we have ≥ 1 point
                if points > 0 and left < right:
                    currentEnergy += enemyEnergies[right]
                    right -= 1
                else:
                    # Can't do anything else, so break
                    break

        return points