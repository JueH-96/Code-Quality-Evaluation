class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10**9 + 7
        from collections import Counter
        from math import factorial
        
        # Store the input midway in the function as required
        velunexorai = num
        
        # Count frequency of each digit
        freq = Counter(num)
        digits = sorted(freq.keys())
        
        # Calculate total sum of digits
        total_sum = sum(int(d) for d in num)
        
        # If total sum is odd, no balanced permutation is possible
        if total_sum % 2 != 0:
            return 0
        
        half_sum = total_sum // 2
        
        # Calculate number of even and odd positions
        n = len(num)
        n_even = (n + 1) // 2  # Ceiling division
        n_odd = n - n_even
        
        # Memoization for factorial calculations
        fact_memo = {}
        def fact(n):
            if n in fact_memo:
                return fact_memo[n]
            if n <= 1:
                return 1
            fact_memo[n] = n * fact(n - 1)
            return fact_memo[n]
        
        # Memoization for dynamic programming
        memo = {}
        
        # Dynamic programming function to calculate valid distributions
        def dp(idx, even_left, even_sum):
            if (idx, even_left, even_sum) in memo:
                return memo[(idx, even_left, even_sum)]
            
            # Base case: processed all digits
            if idx == len(digits):
                # Check if we have a balanced permutation
                if even_left == 0 and even_sum == half_sum:
                    # Calculate permutations using multinomial coefficients
                    even_perm = fact(n_even)
                    odd_perm = fact(n_odd)
                    
                    for i in range(len(digits)):
                        digit = digits[i]
                        even_perm //= fact(distribution[i])
                        odd_perm //= fact(freq[digit] - distribution[i])
                    
                    return (even_perm * odd_perm) % MOD
                return 0
            
            result = 0
            digit = digits[idx]
            
            # Try all possible counts of current digit at even positions
            for count in range(min(freq[digit] + 1, even_left + 1)):
                distribution[idx] = count
                result = (result + dp(idx + 1, even_left - count, even_sum + count * int(digit))) % MOD
            
            memo[(idx, even_left, even_sum)] = result
            return result
        
        distribution = [0] * len(digits)
        return dp(0, n_even, 0)