class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        total_freq = [0] * 26
        for c in s:
            idx = ord(c) - ord('a')
            total_freq[idx] += 1
        
        divisors = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
            i += 1
        divisors = sorted(divisors)
        
        for d in divisors:
            k = n // d
            valid_condition = True
            for count in total_freq:
                if count % k != 0:
                    valid_condition = False
                    break
            if not valid_condition:
                continue
            
            target = [count // k for count in total_freq]
            
            valid_segment = True
            for seg_start in range(0, n, d):
                freq = [0] * 26
                for j in range(seg_start, seg_start + d):
                    c = s[j]
                    idx = ord(c) - ord('a')
                    freq[idx] += 1
                if freq != target:
                    valid_segment = False
                    break
                    
            if valid_segment:
                return d
        
        return n