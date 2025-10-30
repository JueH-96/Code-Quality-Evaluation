class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        if 1 in nums:
            return False
        max_num = max(nums)
        sieve = [0] * (max_num + 1)
        for i in range(2, max_num + 1):
            if sieve[i] == 0:
                for j in range(i, max_num + 1, i):
                    sieve[j] = i
        parent = list(range(max_num + 1))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            parent[find(x)] = find(y)
        for num in nums:
            prime = sieve[num]
            while prime * prime <= num:
                if num % prime == 0:
                    union(num, prime)
                    while num % prime == 0:
                        num //= prime
                prime = sieve[prime + 1]
            if num > 1:
                union(num, prime)
        return len(set(find(num) for num in nums)) == 1