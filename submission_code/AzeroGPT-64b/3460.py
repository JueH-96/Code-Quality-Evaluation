from math import factorial
from typing import List

def mod_inverse(a, m):
    return pow(a, m - 2, m)

def stirling2nd(n, k):
    """
    Calculate the Stirling numbers of the second kind (S(n, k)) using a memoization technique.
    """
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if n == k or k == 1:
        return 1
    return k * stirling2nd(n - 1, k) + stirling2nd(n - 1, k - 1)

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10 ** 9 + 7
        inv_table = [mod_inverse(i, MOD) for i in range(n + 1)]
        self.inv_fact = [1, inv_table[1]]
        for i in range(2, n + 1):
            self.inv_fact.append(self.inv_fact[-1] * inv_table[i] % MOD)
        self.fact = [1] * (n + 1)
        for i in range(1, n + 1):
            self.fact[i] = self.fact[i-1] * i % MOD

        def choose(n, m):
            return self.fact[n] * self.inv_fact[m] * self.inv_fact[n-m] % MOD

        def count_preprefix(n, k):
            """
            Calculate the number of permutations with exactly k inversions and return modulo 10^9 + 7.
            """
            if n == k == 0:
                return 1
            if n == 0 or k < 0:
                return 0
            if n == 1:
                return 1

            count = 0
            for subset_size in range(1, n):
                pre = stirling2nd(k + 1, subset_size + 1)
                suf = count_preprefix(n - subset_size, k - (subset_size * (subset_size + 1) // 2))
                comb = choose(n-1, subset_size)
                count += pre * suf * comb % MOD
            count += stirling2nd(k + 1, n + 1)

            return count % MOD

        # Initialize prefix array with the last requirement
        prefix = [0] * n
        last_requirement = requirements[-1]
        prefix[tuple(last_requirement)] = 1

        # Process each requirement from end to end
        for i in range(len(requirements) - 2, -1, -1):
            end, cnt = requirements[i]
            next_mix = count_preprefix(end + 1, cnt)
            for j in range(end + 1):
                if prefix[(j, prefix[j])]:
                    next_mix *= (factorial(j) * factorial(end - j)) % MOD
                    next_mix *= inv_table[prefix[j]]
                    next_mix *= inv_table[end - prefix[j]]
                    next_mix %= MOD
                    prefix[j] += 1
                if len([1 for x in list(zip(*prefix))][0]) <= 0: break
            prefix[(end, cnt)] = next_mix

        return prefix[(n-1, 0)]