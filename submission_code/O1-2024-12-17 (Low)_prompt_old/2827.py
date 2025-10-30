class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        # Special case: If there's only one element, there's nothing to traverse
        if len(nums) == 1:
            return True
        
        # If any number is 1 and array has more than one element, return False immediately
        # (since gcd(1, x) = 1 which is not > 1, it connects to nobody)
        if any(n == 1 for n in nums):
            # If the array length is > 1, it's impossible to connect 1 with other numbers
            if len(nums) > 1:
                return False
        
        # Precompute smallest prime factors (spf) for each number up to 10^5 
        # to speed up factorization.
        MAX_VAL = 10**5
        spf = [0] * (MAX_VAL+1)   # spf[x] will hold the smallest prime factor of x
        def sieve_spf():
            spf[1] = 1
            for i in range(2, MAX_VAL+1):
                if spf[i] == 0:   # i is prime
                    spf[i] = i
                    for j in range(i*i, MAX_VAL+1, i):
                        if spf[j] == 0:
                            spf[j] = i
        
        sieve_spf()
        
        # DSU (Disjoint Set Union) to merge indices that share a prime factor
        parent = list(range(len(nums)))
        rank = [0] * len(nums)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(a, b):
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
        
        # A map from prime_factor -> an index that has this prime factor
        prime_to_index = {}
        
        # Factor each number, and union current index with any previous index
        # that shares at least one prime factor.
        def factorize_and_union(num, idx):
            # Factorize using spf array
            while num > 1:
                p = spf[num]
                # Union current index with the index that has the same prime factor p
                if p in prime_to_index:
                    union(idx, prime_to_index[p])
                else:
                    prime_to_index[p] = idx
                
                # Move to the next factor
                # Important to skip all occurrences of the same prime factor
                while num % p == 0:
                    num //= p
        
        # Process each number in nums
        for i, val in enumerate(nums):
            factorize_and_union(val, i)
        
        # Check if all indices belong to the same connected component
        # i.e., they have the same root in DSU
        root_set = set(find(i) for i in range(len(nums)))
        return len(root_set) == 1