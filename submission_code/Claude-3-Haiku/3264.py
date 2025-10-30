class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        points = 0
        marked = set()
        
        while True:
            # Find the first unmarked enemy that can be defeated
            for i, energy in enumerate(enemyEnergies):
                if i not in marked and currentEnergy >= energy:
                    # Defeat the enemy and gain 1 point
                    currentEnergy -= energy
                    points += 1
                    break
            else:
                # No more unmarked enemies that can be defeated
                break
            
            # Try to gain energy from marked enemies
            for i, energy in enumerate(enemyEnergies):
                if i in marked and currentEnergy < energy:
                    # Gain energy from the marked enemy
                    currentEnergy += energy
                    marked.remove(i)
                    break
            else:
                # No more marked enemies that can be used to gain energy
                break
        
        return points