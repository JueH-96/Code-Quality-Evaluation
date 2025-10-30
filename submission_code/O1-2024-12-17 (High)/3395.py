class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import Counter
        
        # Count the characters in s
        freq = Counter(s)
        n = len(s)
        
        # A helper function to find all divisors of n in ascending order
        def find_divisors(num):
            divs = []
            i = 1
            while i*i <= num:
                if num % i == 0:
                    divs.append(i)
                    if i != num // i:
                        divs.append(num // i)
                i += 1
            divs.sort()
            return divs
        
        # The idea:
        # If s can be expressed as k copies of some string t (in an anagrammatic way),
        # then length(t) = d and k = n / d.
        # For each character c, freq[c] must be divisible by k. 
        # So we check each divisor d of n in ascending order and see if 
        # for k = n/d, freq[c] % k == 0 for all c in s.
        
        for d in find_divisors(n):
            k = n // d
            if all(ct % k == 0 for ct in freq.values()):
                return d
        
        # If no smaller divisor works, the answer must be n (worst case).
        return n