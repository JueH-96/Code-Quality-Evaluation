class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def find(parent, i):
            if parent[i] == i:
                return i
            parent[i] = find(parent, parent[i])
            return parent[i]

        def union(parent, rank, i, j):
            root_i = find(parent, i)
            root_j = find(parent, j)
            if root_i != root_j:
                if rank[root_i] < rank[root_j]:
                    parent[root_i] = root_j
                elif rank[root_i] > rank[root_j]:
                    parent[root_j] = root_i
                else:
                    parent[root_j] = root_i
                    rank[root_i] += 1

        n = len(nums)
        if n == 1:
            return True

        max_element = max(nums)
        parent = list(range(max_element + 1))
        rank = [0] * (max_element + 1)

        for num in nums:
            if num == 1:
                return False
            d = 2
            temp = num
            while d * d <= temp:
                if temp % d == 0:
                    union(parent, rank, num, d)
                    while temp % d == 0:
                        temp //= d
                d += 1
            if temp > 1:
                union(parent, rank, num, temp)

        
        first_root = -1
        for num in nums:
            root = find(parent, num)
            if first_root == -1:
                first_root = root
            elif first_root != root:
                return False
        return True