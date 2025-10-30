class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 1:
            return True

        def gcd(a, b):
            if b == 0:
                return a
            return gcd(b, a % b)

        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(nums[i], nums[j]) > 1:
                    adj[i].append(j)
                    adj[j].append(i)

        def dfs(start, visited, current_component):
            visited[start] = True
            current_component.append(start)
            for neighbor in adj[start]:
                if not visited[neighbor]:
                    dfs(neighbor, visited, current_component)

        visited = [False] * n
        connected_components = []
        for i in range(n):
            if not visited[i]:
                current_component = []
                dfs(i, visited, current_component)
                connected_components.append(current_component)

        return len(connected_components) == 1