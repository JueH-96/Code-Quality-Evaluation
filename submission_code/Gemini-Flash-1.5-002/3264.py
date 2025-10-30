class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        n = len(enemyEnergies)
        points = 0
        marked = [False] * n
        
        sorted_enemies = sorted(range(n), key=lambda i: enemyEnergies[i])

        
        while True:
            best_enemy = -1
            for i in sorted_enemies:
                if not marked[i] and currentEnergy >= enemyEnergies[i]:
                    best_enemy = i
                    break
            
            if best_enemy == -1:
                break

            currentEnergy -= enemyEnergies[best_enemy]
            points += 1
            
            
            if points > 0:
                best_recharge = -1
                min_energy_unmarked = float('inf')
                for i in sorted_enemies:
                    if not marked[i]:
                        if enemyEnergies[i] < min_energy_unmarked:
                            min_energy_unmarked = enemyEnergies[i]
                            best_recharge = i

                if best_recharge != -1:
                    currentEnergy += enemyEnergies[best_recharge]
                    marked[best_recharge] = True

        return points