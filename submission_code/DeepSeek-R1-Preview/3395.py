class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        
        # Compute all divisors of n
        divisors = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        divisors = sorted(divisors)
        
        # Check each divisor k
        for k in divisors:
            m = n // k
            # Compute target character count for the first chunk
            target = [0] * 26
            for c in s[:k]:
                target[ord(c) - ord('a')] += 1
            
            valid = True
            for i in range(1, m):
                current_count = [0] * 26
                current_chunk = s[i*k : (i+1)*k]
                for c in current_chunk:
                    current_count[ord(c) - ord('a')] += 1
                if current_count != target:
                    valid = False
                    break
            if valid:
                return k
        
        return n  # This line is theoretically unreachable