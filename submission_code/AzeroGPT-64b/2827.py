from collections import defaultdict

def find(a, parent):
    if parent[a] != a:
        parent[a] = find(parent[a], parent)
    return parent[a]

def union(a, b, parent):
    a = find(a, parent)
    b = find(b, parent)
    if a != b:
        parent[a] = b

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) <= 1:
            return True
        
        # Using Sieve of Eratosthenes to find all prime factors
        is_prime = [True] * (max(nums) + 1)
        is_prime[0], is_prime[1] = False, False
        prime_factors = defaultdict(set)
        
        for i in range(2, len(is_prime)):
            if is_prime[i]:
                for j in range(2 * i, len(is_prime), i):
                    is_prime[j] = False
                    prime_factors[j].add(i)
            prime_factors[i].add(i)
        
        prime_count = defaultdict(list)
        idx = dict()  # prime_number -> index

        # Group by prime factors and create index mapping
        for i, num in enumerate(nums):
            primes = prime_factors[num]
            for p in primes:
                prime_count[p].append(i)
                idx[p] = len(prime_count[p]) - 1
        
        # Union entries with same prime factor
        parent = {i: i for i, _ in enumerate(nums)}  # i -> parent[i]
        for p, indices in prime_count.items():
            n = len(indices)
            for i in range(n):
                union(indices[i], indices[0], parent)

        group_size = defaultdict(lambda: 0)
        # Count group size and check for > 1 distinct elements in each group
        for i, _ in enumerate(nums):
            parent[i] = find(i, parent)
            group_size[parent[i]] += 1
        
        return all(g > 1 for g in group_size.values())