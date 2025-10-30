class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        marked = [False] * len(enemyEnergies)
        
        i = 0
        while i < len(enemyEnergies):
            if currentEnergy >= enemyEnergies[i]:
                currentEnergy -= enemyEnergies[i]
                points += 1
                i += 1
            else:
                if points > 0:
                    
                    best_mark_idx = -1
                    best_mark_energy = -1
                    for j in range(len(enemyEnergies)):
                        if marked[j] == False:
                            if best_mark_idx == -1 or enemyEnergies[j] > best_mark_energy:
                                best_mark_idx = j
                                best_mark_energy = enemyEnergies[j]
                    
                    if best_mark_idx != -1:
                        currentEnergy += enemyEnergies[best_mark_idx]
                        marked[best_mark_idx] = True
                    else:
                        break
                else:
                    break

        
        return points