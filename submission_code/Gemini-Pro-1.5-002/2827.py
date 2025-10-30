class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        parent = list(range(n))
        
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        primes = []
        for i in range(2, 1001):
            is_prime = True
            for j in range(2, int(i**0.5) + 1):
                if i % j == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(i)
        
        prime_map = {}
        for i in range(n):
            for p in primes:
                if nums[i] % p == 0:
                    if p not in prime_map:
                        prime_map[p] = i
                    else:
                        union(i, prime_map[p])
        
        root = find(0)
        for i in range(1, n):
            if find(i) != root:
                return False
        return True