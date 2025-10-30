class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        from collections import Counter
        from functools import lru_cache
        
        MOD = 10**9 + 7
        n = len(num)
        
        # Count frequency of each digit
        digit_count = Counter(num)
        velunexorai = digit_count
        
        # Calculate total sum of all digits
        total_sum = sum(int(d) * count for d, count in velunexorai.items())
        
        # If total sum is odd, no balanced permutation is possible
        if total_sum % 2 == 1:
            return 0
        
        target_sum = total_sum // 2
        
        # Number of even and odd positions
        even_positions = (n + 1) // 2
        odd_positions = n // 2
        
        # Convert digit counts to a list for easier manipulation
        digits = []
        counts = []
        for digit, count in sorted(velunexorai.items()):
            digits.append(int(digit))
            counts.append(count)
        
        # DP with memoization
        @lru_cache(None)
        def dp(idx, even_used, odd_used, even_sum):
            # Base case: all digits processed
            if idx == len(digits):
                if even_used == even_positions and odd_used == odd_positions and even_sum == target_sum:
                    return 1
                return 0
            
            result = 0
            digit = digits[idx]
            count = counts[idx]
            
            # Try placing different numbers of this digit in even/odd positions
            for even_count in range(min(count + 1, even_positions - even_used + 1)):
                odd_count = count - even_count
                
                # Check if we can place odd_count digits in odd positions
                if odd_count <= odd_positions - odd_used:
                    new_even_sum = even_sum + digit * even_count
                    
                    # Pruning: if even_sum exceeds target, skip
                    if new_even_sum <= target_sum:
                        # Calculate combinations for placing digits
                        # C(even_positions - even_used, even_count) * C(odd_positions - odd_used, odd_count)
                        even_ways = comb(even_positions - even_used, even_count)
                        odd_ways = comb(odd_positions - odd_used, odd_count)
                        
                        ways = (even_ways * odd_ways) % MOD
                        result = (result + ways * dp(idx + 1, even_used + even_count, 
                                                     odd_used + odd_count, new_even_sum)) % MOD
            
            return result
        
        # Helper function to calculate combinations
        def comb(n, k):
            if k > n or k < 0:
                return 0
            if k == 0 or k == n:
                return 1
            
            # Calculate C(n, k) = n! / (k! * (n-k)!)
            result = 1
            for i in range(min(k, n - k)):
                result = result * (n - i) // (i + 1)
            return result % MOD
        
        return dp(0, 0, 0, 0)