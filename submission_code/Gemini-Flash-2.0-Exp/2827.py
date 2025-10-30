class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True

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
                return True
            return False

        def get_prime_factors(num):
            factors = set()
            d = 2
            while d * d <= num:
                if num % d == 0:
                    factors.add(d)
                    while num % d == 0:
                        num //= d
                d += 1
            if num > 1:
                factors.add(num)
            return factors

        prime_to_index = {}
        for i in range(n):
            factors = get_prime_factors(nums[i])
            for factor in factors:
                if factor in prime_to_index:
                    union(i, prime_to_index[factor])
                else:
                    prime_to_index[factor] = i

        count = 0
        for i in range(n):
            if parent[i] == i:
                count += 1

        return count == 1