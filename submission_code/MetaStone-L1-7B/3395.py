class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Precompute prefix sums for each character
        prefix = {}
        for c in 'abcdefghijklmnopqrstuvwxyz':
            prefix[c] = [0] * (n + 1)
            for i in range(n):
                prefix[c][i+1] = prefix[c][i]
                current_char = s[i]
                prefix[current_char][i+1] += 1
        
        # Get all divisors of n in increasing order
        divisors = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        divisors = sorted(divisors)
        
        for k in divisors:
            if k == 0:
                continue
            
            # Compute the frequency of the first substring of length k
            first_freq = [0] * 26
            for c in range(26):
                c_char = chr(ord('a') + c)
                first_freq[c] = prefix[c_char][k] - prefix[c_char][0]
            
            m = n // k
            valid = True
            
            for i in range(1, m):
                start = i * k
                end = start + k
                current_freq = [0] * 26
                for c in range(26):
                    c_char = chr(ord('a') + c)
                    current_freq[c] = prefix[c_char][end] - prefix[c_char][start]
                if current_freq != first_freq:
                    valid = False
                    break
            
            if valid:
                return k
        
        return n