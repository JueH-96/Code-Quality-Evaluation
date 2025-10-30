from typing import List
import heapq

class Solution:
    def maximumPoints(self, enemyEnergies: List[int], currentEnergy: int) -> int:
        max_points = 0
        heap = []
        for enemy in enemyEnergies:
            if enemy <= currentEnergy:
                heapq.heappush(heap, -enemy)
                currentEnergy -= enemy
                max_points += 1
            elif heap and -heap[0] > enemy:
                max_energy = -heapq.heappop(heap)
                heapq.heappush(heap, -enemy)
                currentEnergy = currentEnergy - enemy + max_energy
                max_points += 1
            else:
                break
        return max_points