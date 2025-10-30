class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        # Precompute smallest prime factors up to 1e5
        max_num = 10**5
        spf = list(range(max_num + 1))
        for i in range(2, int(max_num**0.5) + 1):
            if spf[i] == i:
                for j in range(i * i, max_num + 1, i):
                    if spf[j] == j:
                        spf[j] = i

        parent = {}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            x_root = find(x)
            y_root = find(y)
            if x_root != y_root:
                parent[y_root] = x_root

        primes = set()

        for num in nums:
            if num == 1:
                continue
            x = num
            factors = set()
            while x != 1:
                p = spf[x]
                factors.add(p)
                while x % p == 0:
                    x //= p
            if len(factors) >= 2:
                primes.update(factors)
                first = next(iter(factors))
                for p in factors:
                    if p != first:
                        union(first, p)

        if not primes:
            return len(nums) == 1
        else:
            primes_list = list(primes)
            root = find(primes_list[0])
            for p in primes_list[1:]:
                if find(p) != root:
                    return False
            return True