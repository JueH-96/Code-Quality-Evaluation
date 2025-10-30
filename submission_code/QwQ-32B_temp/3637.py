class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        n = len(num)
        
        # Precompute factorials and inverse factorials up to 80
        max_fact = 80
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
        for i in range(max_fact -1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        # Compute frequency of each digit
        from collections import defaultdict
        freq = defaultdict(int)
        for c in num:
            freq[int(c)] += 1
        
        digits = list(freq.keys())
        total_sum = sum(d * cnt for d, cnt in freq.items())
        
        if total_sum % 2 != 0:
            return 0
        
        target = total_sum // 2
        k = (n + 1) // 2
        m = n // 2
        
        # Initialize DP: dictionary of (sum, count) -> value
        dp = {(0, 0): 1}
        
        for d in digits:
            f = freq[d]
            new_dp = defaultdict(int)
            for (s, c), val in dp.items():
                for x in range(0, f + 1):
                    new_s = s + x * d
                    new_c = c + x
                    
                    # Compute term: C(f, x) / (x! * (f -x)! ) mod MOD
                    # C(f, x) = fact[f] * inv_fact[x] * inv_fact[f-x]
                    C = fact[f] * inv_fact[x] % MOD
                    C = C * inv_fact[f - x] % MOD
                    denom = (fact[x] * fact[f - x]) % MOD
                    inv_denom = pow(denom, MOD-2, MOD)
                    term = C * inv_denom % MOD
                    
                    new_val = val * term % MOD
                    new_dp[(new_s, new_c)] = (new_dp[(new_s, new_c)] + new_val) % MOD
            dp = new_dp
        
        if (target, k) in dp:
            ans = dp[(target, k)] * fact[k] % MOD
            ans = ans * fact[m] % MOD
            return ans
        else:
            return 0