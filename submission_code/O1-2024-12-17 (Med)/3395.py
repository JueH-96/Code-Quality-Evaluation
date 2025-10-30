class Solution:
    def minAnagramLength(self, s: str) -> int:
        from collections import defaultdict
        
        n = len(s)
        # Edge case: if length of s == 1, the answer is 1
        if n == 1:
            return 1
        
        # Build prefix frequency array
        # prefix_freq[i][c] will store how many times character c appears
        # in s[:i], for i in [0..n], c in [0..25].
        prefix_freq = [[0]*26 for _ in range(n+1)]
        
        for i, ch in enumerate(s):
            for c in range(26):
                prefix_freq[i+1][c] = prefix_freq[i][c]
            prefix_freq[i+1][ord(ch) - ord('a')] += 1
        
        # Function to check if a given d is valid
        # i.e., all blocks of length d have the same frequency distribution.
        def valid(d):
            # Frequency distribution of the first block
            base_freq = [prefix_freq[d][c] - prefix_freq[0][c] for c in range(26)]
            
            # Compare frequency distribution for all subsequent blocks
            for start in range(d, n, d):
                block_freq = [prefix_freq[start + d][c] - prefix_freq[start][c] for c in range(26)]
                if block_freq != base_freq:
                    return False
            return True
        
        # Get all divisors of n in ascending order
        divisors = []
        i = 1
        while i*i <= n:
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
            i += 1
        divisors.sort()
        
        # Check divisors from smallest to largest
        for d in divisors:
            if valid(d):
                return d
        
        # Fallback (theoretically should never reach here if everything is correct)
        return n