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
        if len(nums) == 1:
            return True
        # Remove duplicates and handle the case where all elements are 1
        unique_nums = list(set(nums))
        if len(unique_nums) == 1 and unique_nums[0] == 1:
            return False
        # Create a mapping from prime factors to indices
        prime_to_indices = defaultdict(list)
        for idx, num in enumerate(unique_nums):
            if num == 1:
                continue
            factors = self.get_prime_factors(num)
            for p in factors:
                prime_to_indices[p].append(idx)
        # Initialize Union-Find structure
        uf = UnionFind(len(unique_nums))
        # Union all indices that share a common prime factor
        for p in prime_to_indices:
            indices = prime_to_indices[p]
            for i in range(1, len(indices)):
                uf.union(indices[0], indices[i])
        # Check if all indices are in the same set
        root = uf.find(0)
        for i in range(1, len(unique_nums)):
            if uf.find(i) != root:
                return False
        return True

    def get_prime_factors(self, x):
        factors = set()
        # Check for 2 separately
        while x % 2 == 0:
            factors.add(2)
            x = x // 2
        # Check for odd factors from 3 to sqrt(x)
        i = 3
        while i * i <= x:
            while x % i == 0:
                factors.add(i)
                x = x // i
            i += 2
        if x > 2:
            factors.add(x)
        return factors