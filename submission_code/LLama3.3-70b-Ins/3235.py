from typing import List
import heapq

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Create a dictionary to store the cost of changing each character
        cost_dict = {}
        for i in range(len(original)):
            if original[i] not in cost_dict:
                cost_dict[original[i]] = {}
            cost_dict[original[i]][changed[i]] = cost[i]

        # Create a dictionary to store the shortest distance from each character to all other characters
        distance = {}
        for char in set(source + target):
            distance[char] = {char: 0}
            queue = [(0, char)]
            while queue:
                dist, curr_char = heapq.heappop(queue)
                for next_char, next_dist in cost_dict.get(curr_char, {}).items():
                    if next_char not in distance[char] or dist + next_dist < distance[char][next_char]:
                        distance[char][next_char] = dist + next_dist
                        heapq.heappush(queue, (dist + next_dist, next_char))

        # Calculate the minimum cost to convert the string source to the string target
        total_cost = 0
        for i in range(len(source)):
            if source[i] != target[i]:
                if target[i] not in distance[source[i]]:
                    return -1
                total_cost += distance[source[i]][target[i]]

        return total_cost