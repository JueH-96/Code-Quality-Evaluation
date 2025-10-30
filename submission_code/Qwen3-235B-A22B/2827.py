class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 1:
            return True
        if any(num == 1 for num in nums):
            return False
        
        max_num = max(nums)
        
        # Sieve of Eratosthenes to compute smallest prime factors (SPF)
        spf = list(range(max_num + 1))
        for i in range(2, int(max_num ** 0.5) + 1):
            if spf[i] == i:
                for j in range(i * i, max_num + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        # DSU (Union-Find) implementation
        class DSU:
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
        
        dsu = DSU()
        
        # Process each number to factorize and union primes
        for num in nums:
            x = num
            primes = set()
            while x != 1:
                p = spf[x]
                primes.add(p)
                while x % p == 0:
                    x //= p
            # Union all primes for this number
            if not primes:
                return False  # This should not happen as num >=2
            primes_list = list(primes)
            for i in range(1, len(primes_list)):
                dsu.union(primes_list[0], primes_list[i])
        
        # Collect all primes from all numbers
        all_primes = set()
        for num in nums:
            x = num
            while x != 1:
                p = spf[x]
                all_primes.add(p)
                while x % p == 0:
                    x //= p
        
        # Check if all primes are connected
        if not all_primes:
            return False
        first_prime = next(iter(all_primes))
        root = dsu.find(first_prime)
        for p in all_primes:
            if dsu.find(p) != root:
                return False
        return True