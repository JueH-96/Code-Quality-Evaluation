import heapq
from collections import defaultdict

class Solution:
    def maximumPoints(self, enemyEnergies: list[int], currentEnergy: int) -> int:
        if not enemyEnergies:
            return 0
        
        min_heap = enemyEnergies.copy()
        heapq.heapify(min_heap)
        max_heap = [-x for x in enemyEnergies]
        heapq.heapify(max_heap)
        to_remove = defaultdict(int)
        
        points = 0
        energy = currentEnergy
        
        while min_heap and energy >= min_heap[0]:
            while min_heap and to_remove.get(min_heap[0], 0) > 0:
                e = heapq.heappop(min_heap)
                to_remove[e] -= 1
            if not min_heap:
                break
            m = min_heap[0]
            count = energy // m
            points += count
            energy %= m
            
            if energy < m and points > 0 and min_heap:
                while max_heap and to_remove.get(-max_heap[0], 0) > 0:
                    e_val = -heapq.heappop(max_heap)
                    to_remove[e_val] -= 1
                if not max_heap:
                    break
                e_max = -heapq.heappop(max_heap)
                to_remove[e_max] += 1
                energy += e_max
            else:
                break
                
        return points