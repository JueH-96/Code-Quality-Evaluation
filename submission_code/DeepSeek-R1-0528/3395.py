class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        total_freq = [0] * 26
        for c in s:
            total_freq[ord(c) - ord('a')] += 1
        
        divisors = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
            i += 1
        divisors.sort()
        
        for k in divisors:
            m = n // k
            skip = False
            for count in total_freq:
                if count % m != 0:
                    skip = True
                    break
            if skip:
                continue
            
            freq0 = [0] * 26
            for c in s[:k]:
                idx = ord(c) - ord('a')
                freq0[idx] += 1
            
            valid = True
            for i in range(1, m):
                start = i * k
                end = start + k
                freq_i = [0] * 26
                for c in s[start:end]:
                    idx = ord(c) - ord('a')
                    freq_i[idx] += 1
                if freq_i != freq0:
                    valid = False
                    break
            
            if valid:
                return k
        
        return n