class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        """
        We interpret the problem as checking whether the undirected graph
        where vertices are indices of nums, and edges exist between i and j
        iff gcd(nums[i], nums[j]) > 1, is fully connected (i.e. one connected component).

        Key observations:
        1) If nums has length 1, we can trivially return True, because there's only one index.
        2) If any element of nums is 1 and the array length > 1, return False immediately,
           because gcd(1, x) = 1 for any x, which is not > 1, so that index cannot connect to any other.
        3) For nums[i] > 1, we can factor each number. If two numbers share a prime factor p,
           they can be connected. We'll union their indices in a Disjoint Set (Union-Find).
           In the end, if all indices belong to the same connected component, return True;
           otherwise, return False.

        We'll use a sieve-based approach to factor numbers up to 10^5 efficiently.
        """

        import sys
        sys.setrecursionlimit(10**7)

        n = len(nums)
        # If there's only one element, it's trivially connected
        if n == 1:
            return True

        # If any element is 1 and n > 1 => automatically not connected
        # (since gcd(1, x) = 1 <= 1 for any x)
        ones_count = sum(1 for x in nums if x == 1)
        if ones_count > 0:
            # If there's more than 1 element in total and ANY 1 present => false
            return False

        max_val = max(nums)
        # Build the smallest prime factor (spf) array up to max_val
        spf = [0] * (max_val + 1)  # spf[x] will store smallest prime factor of x
        def build_spf(limit):
            for i in range(2, limit+1):
                spf[i] = i
            for i in range(2, int(limit**0.5) + 1):
                if spf[i] == i:  # i is prime
                    for j in range(i*i, limit+1, i):
                        if spf[j] == j:
                            spf[j] = i

        build_spf(max_val)

        # Disjoint set (Union-Find) implementation
        parent = list(range(n))
        rank = [0]*n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                if rank[ra] < rank[rb]:
                    parent[ra] = rb
                elif rank[ra] > rank[rb]:
                    parent[rb] = ra
                else:
                    parent[rb] = ra
                    rank[ra] += 1

        # prime_to_index[p] will store some index that already had prime p
        prime_to_index = {}

        # Function to factor a number using the spf array
        def get_prime_factors(num):
            factors = set()
            while num > 1:
                factors.add(spf[num])
                num //= spf[num]
            return factors

        # For each number, factor it and union indices that share any prime factor
        for i, val in enumerate(nums):
            # Get prime factors
            pf = get_prime_factors(val)
            for p in pf:
                if p not in prime_to_index:
                    prime_to_index[p] = i
                else:
                    union(i, prime_to_index[p])

        # Finally, check if all indices are in the same connected component
        # We can compare their roots to the root of the first index
        root0 = find(0)
        for i in range(1, n):
            if find(i) != root0:
                return False
        return True