from typing import List

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        # We interpret the “complete subset” not based on the values of nums,
        # but rather on the indices. The problem says “a subset of the indices set {1,2,…,n} is complete
        # if the product of every pair of its (indices) is a perfect square.”
        # For two positive integers i and j, i*j is a perfect square if and only if 
        # the “squarefree part” of i (and of j) is the same.
        #
        # (A quick explanation: Any positive integer can be written as (squarefree part) * (perfect square).
        # For i = d * a² and j = d * b² the product i*j = d²*(a*b)² is a perfect square; 
        # conversely, if i and j do not share the same squarefree part then in i*j some prime will be left with an odd exponent.)
        #
        # Thus, the indices that all have the same squarefree part form a complete subset.
        # The “element‐sum” of a complete subset is the sum of nums[i] for every index i in that subset.
        # Our task is to choose the complete subset (i.e. a collection of indices that share the same squarefree part)
        # whose corresponding nums–values have maximum total.
        #
        # For example, in the first test:
        #   nums = [8,7,3,5,7,2,4,9]  (with indices 1,2,…,8)
        # We compute the squarefree part for each index:
        #   1 → 1 
        #   2 → 2
        #   3 → 3
        #   4 = 2² → 1
        #   5 → 5
        #   6 = 2·3 → 2·3 = 6
        #   7 → 7
        #   8 = 2³ → 2   (since 2³ gives odd exponent of 2)
        # Then grouping by the squarefree part gives:
        #    key 1: indices 1 and 4  → sum = nums[0] + nums[3] = 8 + 5 = 13,
        #    key 2: indices 2 and 8  → sum = nums[1] + nums[7] = 7 + 9 = 16,
        #    key 3: index 3          → 3,
        #    key 5: index 5          → 7,
        #    key 6: index 6          → 2,
        #    key 7: index 7          → 4.
        # The maximum sum among these is 16.
        #
        # The same idea works for the second example.
        #
        # Because the “indices” come from 1 to n and n <= 10^4, we can precompute the squarefree part
        # for every integer from 1 to n using a simple sieve (computing the smallest prime factor, spf).
        
        n = len(nums)
        # Precompute smallest prime factors (spf) for numbers 1..n.
        # (n is at most 10^4 so a standard sieve is fast enough.)
        spf = list(range(n + 1))
        for i in range(2, int(n**0.5) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i * i, n + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        # Given x, compute its squarefree part.
        # The squarefree part is the product of each prime factor that appears with odd exponent.
        def squarefree(x: int) -> int:
            res = 1
            while x > 1:
                p = spf[x]
                cnt = 0
                while x % p == 0:
                    cnt += 1
                    x //= p
                if cnt % 2 == 1:
                    res *= p
            return res
        
        # Group the indices (remember: the given array is 1-indexed conceptually).
        # For each index i (from 1 to n), add nums[i-1] to the group corresponding to squarefree(i).
        groups = {}
        for i in range(1, n + 1):
            key = squarefree(i)
            groups[key] = groups.get(key, 0) + nums[i - 1]
        
        return max(groups.values()) if groups else 0