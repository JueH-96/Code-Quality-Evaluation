class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        def gcd(a, b):
            if b == 0:
                return a
            return gcd(b, a % b)

        def lcm(a, b):
            return (a * b) // gcd(a, b)

        n = len(nums)
        parent = list(range(n))
        rank = [0] * n

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                if rank[root_i] < rank[root_j]:
                    parent[root_i] = root_j
                elif rank[root_i] > rank[root_j]:
                    parent[root_j] = root_i
                else:
                    parent[root_j] = root_i
                    rank[root_i] += 1

        num_map = {val: i for i, val in enumerate(nums)}
        
        for i in range(n):
            for j in range(i + 1, n):
                if lcm(nums[i], nums[j]) <= threshold:
                    union(i, j)

        count = 0
        for i in range(n):
            if parent[i] == i:
                count += 1
        return count