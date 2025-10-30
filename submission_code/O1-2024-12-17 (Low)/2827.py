class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        # If there's only one element, we can trivially traverse (nothing to connect).
        if len(nums) == 1:
            return True
        
        # If any element is 1 (and there's more than one element), we cannot connect it to others
        # because gcd(1, x) = 1 (not > 1). Hence return False immediately.
        if any(num == 1 for num in nums):
            return False
        
        # Union-Find / Disjoint Set data structure
        parent = list(range(len(nums)))
        rank = [0]*len(nums)
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(a: int, b: int) -> None:
            rootA = find(a)
            rootB = find(b)
            if rootA != rootB:
                if rank[rootA] < rank[rootB]:
                    parent[rootA] = rootB
                elif rank[rootA] > rank[rootB]:
                    parent[rootB] = rootA
                else:
                    parent[rootB] = rootA
                    rank[rootA] += 1
        
        # Precompute primes up to 100000 using a simple sieve
        max_val = 10**5
        sieve = [True]*(max_val+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_val**0.5)+1):
            if sieve[i]:
                for j in range(i*i, max_val+1, i):
                    sieve[j] = False
        primes = [i for i in range(2, max_val+1) if sieve[i]]
        
        # Dictionary to map prime_factor -> first index that had this factor
        prime_to_index = {}
        
        # Function to factorize a number using the precomputed primes
        def factorize(num: int) -> List[int]:
            factors = []
            tmp = num
            for p in primes:
                if p * p > tmp:
                    break
                if tmp % p == 0:
                    factors.append(p)
                    while tmp % p == 0:
                        tmp //= p
            # If tmp > 1 here, tmp itself is prime
            if tmp > 1:
                factors.append(tmp)
            return factors
        
        # For each number, factorize and union indices that share prime factors
        for i, val in enumerate(nums):
            fcts = factorize(val)
            for p in fcts:
                if p not in prime_to_index:
                    prime_to_index[p] = i
                else:
                    # Union current index with the one that shares prime p
                    union(i, prime_to_index[p])
        
        # Finally, check if all indices are connected (all have the same root)
        root0 = find(0)
        for i in range(1, len(nums)):
            if find(i) != root0:
                return False
        return True