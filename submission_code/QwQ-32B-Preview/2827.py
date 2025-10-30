from typing import List

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        
        if xroot == yroot:
            return
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1

def smallest_prime_factor(n):
    SPF = [0] * (n + 1)
    for i in range(2, n + 1):
        if SPF[i] == 0:
            SPF[i] = i
            for j in range(i * 2, n + 1, i):
                if SPF[j] == 0:
                    SPF[j] = i
    return SPF

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        N = len(nums)
        if N == 1:
            return True
        if 1 in nums and N > 1:
            return False
        
        MAX_NUM = max(nums)
        SPF = smallest_prime_factor(MAX_NUM)
        
        uf = UnionFind(N)
        factor_index = {}
        
        for i, num in enumerate(nums):
            x = num
            while x > 1:
                p = SPF[x]
                if p in factor_index:
                    uf.union(i, factor_index[p])
                else:
                    factor_index[p] = i
                while x % p == 0:
                    x //= p
        
        root = uf.find(0)
        for i in range(1, N):
            if uf.find(i) != root:
                return False
        return True