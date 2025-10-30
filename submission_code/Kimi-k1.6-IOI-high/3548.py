import math

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        if n == 0:
            return 0
        
        m = (n + 1) // 2
        start = 10 ** (m - 1)
        end = (10 ** m) - 1
        seen = set()
        
        for num in range(start, end + 1):
            s = str(num)
            if n % 2 == 0:
                pal_str = s + s[::-1]
            else:
                pal_str = s + s[:-1][::-1]
            palindrome = int(pal_str)
            if palindrome % k != 0:
                continue
            
            # Compute digit counts
            counts = [0] * 10
            for c in pal_str:
                counts[int(c)] += 1
            counts_tuple = tuple(counts)
            seen.add(counts_tuple)
        
        # Precompute factorials up to n
        fact = [math.factorial(i) for i in range(n + 1)]
        total = 0
        
        for counts in seen:
            numerator = fact[n]
            denominator = 1
            for cnt in counts:
                if cnt > 0:
                    denominator *= math.factorial(cnt)
            total_perm = numerator // denominator
            
            if counts[0] == 0:
                invalid = 0
            else:
                numerator_invalid = fact[n-1]
                denominator_invalid = math.factorial(counts[0] - 1)
                for i in range(1, 10):
                    denominator_invalid *= math.factorial(counts[i])
                invalid = numerator_invalid // denominator_invalid
            
            valid = total_perm - invalid
            total += valid
        
        return total