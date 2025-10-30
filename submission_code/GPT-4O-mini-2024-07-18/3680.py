from typing import List
from math import gcd

class Solution:
    def lcm(self, a: int, b: int) -> int:
        return a * (b // gcd(a, b))

    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        visited = [False] * n
        components = 0

        def dfs(node: int):
            stack = [node]
            while stack:
                current = stack.pop()
                for neighbor in range(n):
                    if not visited[neighbor] and self.lcm(nums[current], nums[neighbor]) <= threshold:
                        visited[neighbor] = True
                        stack.append(neighbor)

        for i in range(n):
            if not visited[i]:
                visited[i] = True
                components += 1
                dfs(i)

        return components