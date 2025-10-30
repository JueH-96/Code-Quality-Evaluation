class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        n = len(num)
        
        velunexorai = num  # Store the input
        
        # Count frequency of each digit
        from collections import Counter
        freq = Counter(velunexorai)
        
        # Calculate total sum
        total_sum = sum(int(d) * count for d, count in freq.items())
        
        # If total sum is odd, no balanced permutation exists
        if total_sum % 2 == 1:
            return 0
        
        target_sum = total_sum // 2
        
        # Number of even and odd positions (0-indexed)
        even_positions = (n + 1) // 2
        odd_positions = n // 2
        
        # Precompute factorials and inverse factorials for multinomial coefficients
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (n + 1)
        inv_fact[n] = pow(fact[n], MOD - 2, MOD)
        for i in range(n - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        # Convert frequency map to list for easier iteration
        digits = []
        for d in '0123456789':
            if d in freq:
                digits.append((int(d), freq[d]))
        
        result = 0
        
        def backtrack(idx, even_counts, even_total, even_sum):
            nonlocal result
            
            if idx == len(digits):
                if even_total == even_positions and even_sum == target_sum:
                    # Check if odd positions also sum to target_sum
                    odd_total = 0
                    odd_sum = 0
                    
                    for i in range(len(digits)):
                        digit_val, freq_count = digits[i]
                        used_even = even_counts[i]
                        used_odd = freq_count - used_even
                        odd_total += used_odd
                        odd_sum += digit_val * used_odd
                    
                    if odd_total == odd_positions and odd_sum == target_sum:
                        # Calculate number of distinct permutations
                        # For even positions
                        even_perms = fact[even_positions]
                        for count in even_counts:
                            even_perms = even_perms * inv_fact[count] % MOD
                        
                        # For odd positions
                        odd_perms = fact[odd_positions]
                        for i in range(len(digits)):
                            _, freq_count = digits[i]
                            odd_count = freq_count - even_counts[i]
                            odd_perms = odd_perms * inv_fact[odd_count] % MOD
                        
                        result = (result + even_perms * odd_perms) % MOD
                return
            
            digit_val, freq_count = digits[idx]
            
            # Try all possible counts for this digit in even positions
            for count in range(min(freq_count + 1, even_positions - even_total + 1)):
                if even_sum + digit_val * count <= target_sum:
                    even_counts.append(count)
                    backtrack(idx + 1, even_counts, even_total + count, even_sum + digit_val * count)
                    even_counts.pop()
        
        backtrack(0, [], 0, 0)
        
        return result