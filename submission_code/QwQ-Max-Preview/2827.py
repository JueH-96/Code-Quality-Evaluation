class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 1
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1

class Solution:
    _spf = None  # Class-level variable for smallest prime factors

    @classmethod
    def precompute_spf(cls):
        if cls._spf is not None:
            return
        max_num = 10**5
        cls._spf = list(range(max_num + 1))
        for i in range(2, int(max_num**0.5) + 1):
            if cls._spf[i] == i:
                for j in range(i*i, max_num + 1, i):
                    if cls._spf[j] == j:
                        cls._spf[j] = i

    def get_primes(self, n):
        primes = set()
        if n == 1:
            return primes
        while n != 1:
            p = self._spf[n]
            primes.add(p)
            while n % p == 0:
                n = n // p
        return primes

    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        self.precompute_spf()

        if len(nums) == 1:
            return True
        if 1 in nums:
            return False
        
        primes_set = set()
        uf = UnionFind()
        
        for num in nums:
            primes = self.get_primes(num)
            primes_list = list(primes)
            if not primes_list:
                continue
            for p in primes:
                primes_set.add(p)
            first = primes_list[0]
            for p in primes_list[1:]:
                uf.union(first, p)
        
        if not primes_set:
            return False
        
        root = uf.find(next(iter(primes_set)))
        for p in primes_set:
            if uf.find(p) != root:
                return False
        
        return True