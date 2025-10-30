from typing import List
from collections import defaultdict
from math import gcd

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def is_perfect_square(x):
            root = int(x**0.5)
            return root * root == x

        def can_form_complete_set(a, b):
            product = a * b
            return is_perfect_square(product)

        n = len(nums)
        max_sum = 0

        # Create a graph where an edge exists between i and j if nums[i] * nums[j] is a perfect square
        graph = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                if can_form_complete_set(nums[i], nums[j]):
                    graph[i].append(j)
                    graph[j].append(i)

        # Use DFS to find all connected components in the graph
        visited = [False] * n

        def dfs(node):
            stack = [node]
            component = []
            while stack:
                u = stack.pop()
                if not visited[u]:
                    visited[u] = True
                    component.append(u)
                    for v in graph[u]:
                        if not visited[v]:
                            stack.append(v)
            return component

        # Find all connected components and calculate their sums
        for i in range(n):
            if not visited[i]:
                component = dfs(i)
                component_sum = sum(nums[idx] for idx in component)
                max_sum = max(max_sum, component_sum)

        return max_sum