class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True
        if 1 in nums:
            return False
        
        max_val = max(nums)
        spf = list(range(max_val + 1))
        
        for i in range(2, int(max_val**0.5) + 1):
            if spf[i] == i:
                for j in range(i*i, max_val + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        class DSU:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
                self.count = n
            
            def find(self, a):
                if self.parent[a] != a:
                    self.parent[a] = self.find(self.parent[a])
                return self.parent[a]
            
            def union(self, a, b):
                aroot = self.find(a)
                broot = self.find(b)
                if aroot == broot:
                    return
                if self.rank[aroot] < self.rank[broot]:
                    aroot, broot = broot, aroot
                self.parent[broot] = aroot
                if self.rank[aroot] == self.rank[broot]:
                    self.rank[aroot] += 1
                self.count -= 1
        
        dsu = DSU(n)
        prime_rep = {}
        
        for i, num in enumerate(nums):
            temp = num
            factors = set()
            while temp > 1:
                p = spf[temp]
                factors.add(p)
                while temp % p == 0:
                    temp //= p
            for p in factors:
                if p in prime_rep:
                    dsu.union(i, prime_rep[p])
                else:
                    prime_rep[p] = i
        
        return dsu.count == 1