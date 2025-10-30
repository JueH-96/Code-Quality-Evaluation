class Solution:
    def minAnagramLength(self, s: str) -> int:
        from math import isqrt
        n = len(s)
        divisors = set()
        for i in range(1, isqrt(n)+1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        divisors = sorted(divisors)
        s_list = list(s)
        for k in divisors:
            if k > n:
                continue
            substrings = [s[i:i+k] for i in range(0, n, k)]
            base_count = [0]*26
            for c in substrings[0]:
                base_count[ord(c)-ord('a')] +=1
            valid = True
            for substr in substrings[1:]:
                curr_count = [0]*26
                for c in substr:
                    curr_count[ord(c)-ord('a')] +=1
                if curr_count != base_count:
                    valid = False
                    break
            if valid:
                return k
        return n