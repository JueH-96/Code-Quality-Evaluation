from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # sort the energies (tokens)
        enemyEnergies.sort()
        i, j = 0, len(enemyEnergies) - 1
        score = 0
        maxScore = 0
        while i <= j:
            # if we have enough energy, "buy" a token face up
            if currentEnergy >= enemyEnergies[i]:
                currentEnergy -= enemyEnergies[i]
                score += 1
                i += 1
                if score > maxScore:
                    maxScore = score
            # otherwise, if we have at least one point, try selling a token face down
            elif score > 0 and i <= j:
                currentEnergy += enemyEnergies[j]
                score -= 1
                j -= 1
            else:
                break
        return maxScore