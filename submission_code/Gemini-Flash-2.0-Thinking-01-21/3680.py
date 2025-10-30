import math

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def lcm(a, b):
            return (a * b) // gcd(a, b)

        n = len(nums)
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if lcm(nums[i], nums[j]) <= threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * n
        components = 0
        for i in range(n):
            if not visited[i]:
                components += 1
                q = [i]
                visited[i] = True
                while q:
                    u = q.pop(0)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            q.append(v)
        return components