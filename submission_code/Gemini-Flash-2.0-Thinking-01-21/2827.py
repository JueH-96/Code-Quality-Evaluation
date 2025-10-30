import math

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 1:
            return True
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(nums[i], nums[j]) > 1:
                    adj[i].append(j)
                    adj[j].append(i)

        for i in range(n):
            for j in range(i + 1, n):
                q = [i]
                visited = {i}
                found_path = False
                while q:
                    u = q.pop(0)
                    if u == j:
                        found_path = True
                        break
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append(v)
                if not found_path:
                    return False
        return True