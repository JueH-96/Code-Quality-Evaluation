class Solution:
    _spf = None  # Class-level variable to store the SPF array
    _max_possible = 10**5

    def __init__(self):
        if Solution._spf is None:
            spf = [0] * (Solution._max_possible + 1)
            for i in range(2, Solution._max_possible + 1):
                if spf[i] == 0:
                    spf[i] = i
                    for j in range(i * i, Solution._max_possible + 1, i):
                        if spf[j] == 0:
                            spf[j] = i
            Solution._spf = spf

    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True
        if 1 in nums:
            return False

        parent = {}
        spf = Solution._spf

        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # Path compression
                u = parent[u]
            return u

        def union(u, v):
            pu = find(u)
            pv = find(v)
            if pu != pv:
                if pu < pv:
                    parent[pv] = pu
                else:
                    parent[pu] = pv

        all_primes = set()

        for num in nums:
            primes = set()
            x = num
            while x > 1:
                p = spf[x]
                primes.add(p)
                x = x // p
            if not primes:
                return False  # This case shouldn't occur as num >= 2

            primes_list = list(primes)
            # Add all primes to the parent dictionary if not present
            for p in primes_list:
                if p not in parent:
                    parent[p] = p

            # Union all primes in the current number's prime factors
            if primes_list:
                first_p = primes_list[0]
                for p in primes_list[1:]:
                    union(first_p, p)
            all_primes.update(primes)

        if not all_primes:
            return False  # This case shouldn't occur

        # Check if all primes are connected
        first_p = next(iter(all_primes))
        for p in all_primes:
            if find(p) != find(first_p):
                return False

        return True