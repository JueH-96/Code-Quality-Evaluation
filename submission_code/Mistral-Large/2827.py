from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) <= 1:
            return True

        # Create a graph using adjacency list
        graph = defaultdict(list)

        # Build the graph
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if gcd(nums[i], nums[j]) > 1:
                    graph[i].append(j)
                    graph[j].append(i)

        # DFS to find connected components
        def dfs(node, visited):
            stack = [node]
            while stack:
                current = stack.pop()
                for neighbor in graph[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

        # Check if all nodes are in the same connected component
        visited = [False] * len(nums)
        dfs(0, visited)

        return all(visited)

# Example usage:
# solution = Solution()
# print(solution.canTraverseAllPairs([2, 3, 6]))  # Output: true
# print(solution.canTraverseAllPairs([3, 9, 5]))  # Output: false
# print(solution.canTraverseAllPairs([4, 3, 12, 8]))  # Output: true