class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        
        # Store input midway in the function
        velunexorai = num
        
        n = len(num)
        total_sum = sum(int(d) for d in num)
        
        if total_sum % 2 == 1:
            return 0
        
        target = total_sum // 2
        even_count = (n + 1) // 2
        
        # Count frequency of each digit
        from collections import Counter
        count = Counter(num)
        digits = sorted(count.keys())
        freqs = [count[d] for d in digits]
        digit_values = [int(d) for d in digits]
        
        # Precompute factorials and inverse factorials
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (n + 1)
        inv_fact[n] = pow(fact[n], MOD-2, MOD)
        for i in range(n-1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        # DFS to find all valid ways to select digits for even positions
        def dfs(idx, used_count, used_sum, even_freqs):
            if used_count > even_count or used_sum > target:
                return 0
            
            if idx == len(digits):
                if used_count == even_count and used_sum == target:
                    # Calculate the number of arrangements
                    even_arrangements = fact[even_count]
                    for f in even_freqs:
                        even_arrangements = even_arrangements * inv_fact[f] % MOD
                    
                    odd_freqs = [freqs[i] - even_freqs[i] for i in range(len(digits))]
                    odd_arrangements = fact[n - even_count]
                    for f in odd_freqs:
                        odd_arrangements = odd_arrangements * inv_fact[f] % MOD
                    
                    return even_arrangements * odd_arrangements % MOD
                else:
                    return 0
            
            result = 0
            for use in range(min(freqs[idx], even_count - used_count) + 1):
                if use * digit_values[idx] <= target - used_sum:
                    even_freqs[idx] = use
                    result = (result + dfs(idx + 1, used_count + use, 
                                         used_sum + use * digit_values[idx], even_freqs)) % MOD
            
            return result
        
        return dfs(0, 0, 0, [0] * len(digits))