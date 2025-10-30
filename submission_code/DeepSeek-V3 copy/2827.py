from math import gcd
from collections import defaultdict

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # Path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False
        if self.size[xroot] < self.size[yroot]:
            xroot, yroot = yroot, xroot
        self.parent[yroot] = xroot
        self.size[xroot] += self.size[yroot]
        return True

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if not nums:
            return False
        n = len(nums)
        if n == 1:
            return True
        uf = UnionFind(n)
        prime_to_index = defaultdict(list)
        for i, num in enumerate(nums):
            if num == 1:
                return False
            factors = self.get_prime_factors(num)
            for p in factors:
                prime_to_index[p].append(i)
        for p in prime_to_index:
            indices = prime_to_index[p]
            for i in range(1, len(indices)):
                uf.union(indices[0], indices[i])
        root = uf.find(0)
        for i in range(1, n):
            if uf.find(i) != root:
                return False
        return True

    def get_prime_factors(self, x):
        factors = set()
        while x % 2 == 0:
            factors.add(2)
            x = x // 2
        i = 3
        while i * i <= x:
            while x % i == 0:
                factors.add(i)
                x = x // i
            i += 2
        if x > 2:
            factors.add(x)
        return factors