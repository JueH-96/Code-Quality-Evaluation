class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        # Precompute factorial up to n
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i
        
        m = (n + 1) // 2
        start = 10 ** (m - 1)
        end = 10 ** m
        
        multisets = set()
        
        # Generate all palindromes and collect unique multisets
        for s in range(start, end):
            s_str = str(s)
            if n % 2 == 0:
                pal_str = s_str + s_str[::-1]
            else:
                pal_str = s_str + s_str[:-1][::-1]
            
            # Check if the palindrome is divisible by k
            pal = int(pal_str)
            if pal % k != 0:
                continue
            
            # Compute the digit counts
            counts = [0] * 10
            for c in pal_str:
                d = int(c)
                counts[d] += 1
            multisets.add(tuple(counts))
        
        total_sum = 0
        
        # Calculate the valid numbers for each multiset
        for counts in multisets:
            denominator = 1
            for c in counts:
                denominator *= fact[c]
            total = fact[n] // denominator
            
            invalid = 0
            if counts[0] > 0:
                new_counts = list(counts)
                new_counts[0] -= 1
                denominator_invalid = 1
                for c in new_counts:
                    denominator_invalid *= fact[c]
                invalid = fact[n - 1] // denominator_invalid
            
            valid = total - invalid
            total_sum += valid
        
        return total_sum