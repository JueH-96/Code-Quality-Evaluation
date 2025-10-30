class Solution:
    def minAnagramLength(self, s: str) -> int:
        def get_divisors(n):
            divisors = set()
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    divisors.add(i)
                    divisors.add(n // i)
            return sorted(divisors, reverse=True)
        
        def get_freq(s_part):
            counts = [0] * 26
            for c in s_part:
                counts[ord(c) - ord('a')] += 1
            return tuple(counts)
        
        n_total = len(s)
        if n_total == 0:
            return 0
        
        total_freq = get_freq(s)
        
        divisors = get_divisors(n_total)
        for d in divisors:
            # Check if all character counts are divisible by d
            valid = True
            for count in total_freq:
                if count % d != 0:
                    valid = False
                    break
            if not valid:
                continue
            
            n = n_total // d
            # Check each part's frequency
            first_freq = get_freq(s[:n])
            valid_parts = True
            for i in range(1, d):
                start = i * n
                end = start + n
                part = s[start:end]
                part_freq = get_freq(part)
                if part_freq != first_freq:
                    valid_parts = False
                    break
            if valid_parts:
                return n
        
        return n_total