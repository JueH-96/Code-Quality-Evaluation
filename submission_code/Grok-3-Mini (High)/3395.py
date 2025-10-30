class Solution:
    def minAnagramLength(self, s: str) -> int:
        len_s = len(s)
        # Find all divisors of len_s
        divisors = []
        for i in range(1, int(len_s**0.5) + 1):
            if len_s % i == 0:
                divisors.append(i)
                if i != len_s // i:
                    divisors.append(len_s // i)
        divisors.sort()
        
        # Helper function to get the frequency signature as a tuple
        def get_sig(sub):
            count = [0] * 26
            for c in sub:
                idx = ord(c) - ord('a')
                count[idx] += 1
            return tuple(count)
        
        # Check each divisor starting from the smallest
        for d in divisors:
            first_sig = get_sig(s[:d])
            if all(get_sig(s[start:start + d]) == first_sig for start in range(0, len_s, d)):
                return d