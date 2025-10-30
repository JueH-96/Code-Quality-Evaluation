from math import factorial

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        if n == 0:
            return 0
        
        def generate_palindromes(n):
            if n == 1:
                return [str(d) for d in range(1, 10)]
            m = (n + 1) // 2
            palindromes = []
            start = 10 ** (m - 1)
            end = 10 ** m
            for prefix in range(start, end):
                s = str(prefix)
                if n % 2 == 0:
                    palindrome_str = s + s[::-1]
                else:
                    palindrome_str = s + s[:-1][::-1]
                palindromes.append(palindrome_str)
            return palindromes
        
        palindromes = generate_palindromes(n)
        
        good_multisets = set()
        
        for p_str in palindromes:
            num = int(p_str)
            if num % k == 0:
                key = ''.join(sorted(p_str))
                good_multisets.add(key)
        
        total = 0
        for key in good_multisets:
            counts = {}
            for d in key:
                counts[d] = counts.get(d, 0) + 1
            
            n_total = len(key)
            total_perms = factorial(n_total)
            for cnt in counts.values():
                total_perms //= factorial(cnt)
            
            if '0' in counts:
                if counts['0'] == n_total:
                    valid = 0
                else:
                    new_counts = counts.copy()
                    new_counts['0'] -= 1
                    if new_counts['0'] == 0:
                        del new_counts['0']
                    
                    invalid_perms = factorial(n_total - 1)
                    for cnt in new_counts.values():
                        invalid_perms //= factorial(cnt)
                    valid = total_perms - invalid_perms
            else:
                valid = total_perms
            
            total += valid
        
        return total