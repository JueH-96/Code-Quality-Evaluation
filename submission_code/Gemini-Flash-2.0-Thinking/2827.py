import math

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 1:
            return True

        uf = UnionFind(n)
        prime_factors_map = {}

        def get_prime_factors(num):
            factors = set()
            d = 2
            temp = num
            while d * d <= temp:
                if temp % d == 0:
                    factors.add(d)
                    while temp % d == 0:
                        temp //= d
                d += 1
            if temp > 1:
                factors.add(temp)
            return factors

        for i in range(n):
            num = nums[i]
            factors = get_prime_factors(num)
            for factor in factors:
                if factor in prime_factors_map:
                    for index in prime_factors_map[factor]:
                        uf.union(i, index)
                    prime_factors_map[factor].add(i)
                else:
                    prime_factors_map[factor] = {i}

        root = uf.find(0)
        for i in range(1, n):
            if uf.find(i) != root:
                return False

        return True