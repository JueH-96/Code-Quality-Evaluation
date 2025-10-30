from collections import defaultdict

class Solution:
    def longestSpecialPath(self, edges: List[List[int]], nums: List[int]) -> List[int]:
        n = len(nums)
        graph = defaultdict(list)
        for u, v, length in edges:
            graph[u].append((v, length))
            graph[v].append((u, length))

        max_length = 0
        min_nodes = float('inf')

        def dfs(node, parent, path):
            nonlocal max_length, min_nodes
            if len(path) > max_length or (len(path) == max_length and len(path) < min_nodes):
                max_length = len(path)
                min_nodes = len(path)

            for neighbor, length in graph[node]:
                if neighbor != parent and nums[neighbor] not in path:
                    dfs(neighbor, node, path + [nums[neighbor]])

        dfs(0, -1, [nums[0]])
        return [max_length, min_nodes]