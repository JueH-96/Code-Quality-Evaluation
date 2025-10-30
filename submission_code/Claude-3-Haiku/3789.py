from typing import List
from collections import defaultdict

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        # Create a graph to represent the conflicting pairs
        graph = defaultdict(list)
        for a, b in conflictingPairs:
            graph[a].append(b)
            graph[b].append(a)
        
        # Find the maximum number of subarrays after removing one conflicting pair
        max_subarrays = 0
        for i, j in conflictingPairs:
            # Remove the current conflicting pair
            graph[i].remove(j)
            graph[j].remove(i)
            
            # Count the number of subarrays that do not contain both i and j
            subarrays = 0
            for num in range(1, n+1):
                if all(num not in graph[x] for x in graph[num]):
                    subarrays += n - len(graph[num]) + 1
            
            # Restore the conflicting pair
            graph[i].append(j)
            graph[j].append(i)
            
            # Update the maximum number of subarrays
            max_subarrays = max(max_subarrays, subarrays)
        
        return max_subarrays