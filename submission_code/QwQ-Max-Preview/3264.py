import heapq
from typing import List

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        enemyEnergies.sort()
        points = 0
        n = len(enemyEnergies)
        # Initial attack phase
        for e in enemyEnergies:
            if currentEnergy < e:
                break
            count = currentEnergy // e
            points += count
            currentEnergy -= count * e
        
        if n == 0:
            return points
        
        # Create a max-heap of all enemies
        max_heap = [-e for e in enemyEnergies]
        heapq.heapify(max_heap)
        recharged = set()
        e_min = enemyEnergies[0]
        found_min = False
        
        while max_heap and points >= 1:
            # Find the largest enemy not yet recharged
            e_max = None
            while max_heap:
                candidate = -heapq.heappop(max_heap)
                if candidate not in recharged:
                    e_max = candidate
                    break
            if e_max is None:
                break  # no available enemies to recharge
            # Recharge this enemy
            points -= 1
            currentEnergy += e_max
            recharged.add(e_max)
            # Now attack the smallest enemy not recharged
            # Find the current minimum enemy not recharged
            if not found_min:
                # Find the first enemy not recharged
                for e in enemyEnergies:
                    if e not in recharged:
                        e_min = e
                        break
                found_min = True
            else:
                # Check if e_min is still valid
                if e_min in recharged:
                    # Find new e_min
                    for e in enemyEnergies:
                        if e not in recharged:
                            e_min = e
                            break
                    if e_min in recharged:
                        break  # no enemies left to attack
            # Attack e_min as many times as possible
            if currentEnergy >= e_min:
                count = currentEnergy // e_min
                points += count
                currentEnergy -= count * e_min
                # Add e_min back to the heap if not already present
                # Since we have recharged set, no need to add back
        
        return points