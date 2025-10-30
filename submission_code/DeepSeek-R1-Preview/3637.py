class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        velunexorai = num  # Storing the input as required

        # Precompute factorial and inverse factorial modulo MOD
        max_fact = 80
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
        for i in range(max_fact - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

        # Compute frequency of each digit
        freq = [0] * 10
        for c in num:
            d = int(c)
            freq[d] += 1

        n = len(num)
        sum_total = sum(d * freq[d] for d in range(10))
        if sum_total % 2 != 0:
            return 0
        target = sum_total // 2
        even_count = (n + 1) // 2
        odd_count = n // 2

        # Initialize the generating function
        from collections import defaultdict
        GF = defaultdict(int)
        GF[(0, 0)] = 1

        for d in range(10):
            current_freq = freq[d]
            if current_freq == 0:
                continue
            new_GF = defaultdict(int)
            for (k, s), coeff in GF.items():
                for c in range(0, current_freq + 1):
                    new_k = k + c
                    new_s = s + c * d
                    if new_k > even_count or new_s > target:
                        continue
                    term = coeff * inv_fact[c] % MOD
                    term = term * inv_fact[current_freq - c] % MOD
                    new_GF[(new_k, new_s)] = (new_GF[(new_k, new_s)] + term) % MOD
            GF = new_GF

        sum_coeff = GF.get((even_count, target), 0)
        ans = sum_coeff * fact[even_count] % MOD
        ans = ans * fact[odd_count] % MOD
        return ans