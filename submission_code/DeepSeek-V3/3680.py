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
    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        uf = UnionFind(n)
        num_to_index = {num: i for i, num in enumerate(nums)}
        
        # Precompute all possible multiples of each number up to the threshold
        for i in range(n):
            num = nums[i]
            if num > threshold:
                continue
            for multiple in range(num, threshold + 1, num):
                if multiple in num_to_index:
                    uf.union(i, num_to_index[multiple])
        
        # Count the number of unique parents
        unique_parents = set()
        for i in range(n):
            unique_parents.add(uf.find(i))
        
        return len(unique_parents)