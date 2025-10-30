from typing import List
import heapq

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        # Sort the enemy energies in ascending order
        enemyEnergies.sort()
        
        points = 0
        n = len(enemyEnergies)
        
        # Use a min-heap to keep track of the smallest energies we can use to gain points
        min_heap = []
        
        for i in range(n):
            if currentEnergy >= enemyEnergies[i]:
                # If we can defeat the enemy, gain a point and reduce current energy
                points += 1
                currentEnergy -= enemyEnergies[i]
                # Push the energy to the heap for potential future use
                heapq.heappush(min_heap, enemyEnergies[i])
            else:
                # If we cannot defeat the enemy, check if we can use a previously defeated enemy's energy
                if min_heap:
                    # Get the smallest energy from the heap
                    smallest_energy = heapq.heappop(min_heap)
                    # If using the smallest energy helps us defeat the current enemy
                    if currentEnergy + smallest_energy >= enemyEnergies[i]:
                        currentEnergy += smallest_energy
                        points += 1
                        currentEnergy -= enemyEnergies[i]
                        heapq.heappush(min_heap, enemyEnergies[i])
                    else:
                        # If not, push the smallest energy back to the heap
                        heapq.heappush(min_heap, smallest_energy)
                else:
                    # If no more enemies can be defeated, break the loop
                    break
        
        # After the loop, we can use the remaining energies in the heap to gain more points
        while min_heap and currentEnergy > 0:
            smallest_energy = heapq.heappop(min_heap)
            if currentEnergy + smallest_energy >= smallest_energy:
                points += 1
                currentEnergy += smallest_energy
                currentEnergy -= smallest_energy
            else:
                break
        
        return points