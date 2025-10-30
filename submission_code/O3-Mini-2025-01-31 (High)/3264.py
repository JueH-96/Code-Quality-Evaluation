from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # The key observation is that “attacking” an enemy (operation 1)
        # does not "use up" an enemy – it never gets marked – so you can use
        # the same enemy as many times as you like provided your current energy
        # is at least its energy requirement.
        #
        # Therefore, to “buy” a point with as little energy waste as possible you
        # always want to choose the enemy with the smallest energy value.
        # In contrast, an enemy used in operation 2 (the “energy boost”) has
        # to be marked and can be used only once. And since operation 2 doesn’t
        # change your point count (it only adds energy) you want to apply it on
        # enemies that give the most energy.
        #
        # The optimal overall strategy is:
        # 1. Reserve one enemy having the minimum energy, m = min(enemyEnergies),
        #    to be used repeatedly for operation 1 (each use gives +1 point at a cost of m energy).
        # 2. If you never have enough energy (i.e. currentEnergy < m) you cannot even
        #    perform a single operation 1 (and you can’t do operation 2 because you need at least 1 point).
        # 3. Otherwise, use exactly 1 operation 1 (on the reserved enemy) to “unlock” operation 2.
        #    This costs m energy and gives you 1 point.
        # 4. Now, use EVERY other enemy (each exactly once) with operation 2.
        #    (Even if some enemies have energy equal to m, using them is never a “loss” because
        #     operation 2 is free in points and merely adds energy.)
        # 5. At that point your total energy available will be:
        #       remaining_energy = (currentEnergy - m)  from the initial energy (after spending m to get 1 point)
        #       plus the sum of all enemy energies from the ones you used in operation 2.
        #    That is, total_energy = (currentEnergy - m) + (sum(enemyEnergies) - m).
        # 6. Since from here you can use the reserved enemy (with cost m) as many times as possible,
        #    the additional points you get will be floor(total_energy / m).
        #
        # Special case: if there is only one enemy, you cannot use operation 2 at all.
        #
        # Thus, if n = len(enemyEnergies) and m = min(enemyEnergies):
        #   • if currentEnergy < m: answer is 0.
        #   • if n == 1: answer = currentEnergy // m.
        #   • Otherwise, answer = 1 + ((currentEnergy - m) + (sum(enemyEnergies)-m)) // m.
        
        n = len(enemyEnergies)
        m = min(enemyEnergies)
        
        # If we can't even "afford" one operation 1, we cannot get any points.
        if currentEnergy < m:
            return 0
        
        # If there's only one enemy, you can only use operation 1 repeatedly.
        if n == 1:
            return currentEnergy // m
        
        total_sum = sum(enemyEnergies)
        # Use one op1 move on the reserved enemy (cost m, +1 point),
        # leaving us with currentEnergy - m energy.
        remaining_energy = currentEnergy - m
        
        # The rest of the enemies can each be spent exactly once with op2.
        # Their total contribution is:
        op2_total = total_sum - m  # (we are keeping one copy of m for op1)
        
        # Now all the energy you have – (remaining_energy + op2_total) – can be used
        # to do as many additional op1 moves as possible. Each op1 move costs m.
        additional_points = (remaining_energy + op2_total) // m
        
        # Final points: 1 (from the initial op1 move) plus all extra op1 moves you can afford.
        return 1 + additional_points