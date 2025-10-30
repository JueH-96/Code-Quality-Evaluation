class Solution:
    def minAnagramLength(self, s: str) -> int:
        """
        The string  s  is the concatenation of several contiguous blocks.
        All those blocks are anagrams of the same (unknown) string  t .
        We have to find the minimum possible length of such a block.
        
        Approach
        --------
        • Let n = len(s).  
        • The length L of every block must divide n – otherwise the blocks
          could not tile the whole string.
        • Therefore we only have to examine every divisor of n and find the
          smallest one that really works.
        
        Checking a candidate length L
        -----------------------------
        1. Count the 26 letters inside the first block (positions 0 … L-1);
           that is the reference frequency vector.
        2. Scan the remaining blocks of length L one after another.
           For every block build its frequency vector and compare it to the
           reference.  
           – If all blocks match, L is valid.  
           – Otherwise L is impossible, move on to the next divisor.
        
        Complexity
        ----------
        • There are at most about 120 divisors when n ≤ 10^5.
        • For every divisor we inspect each character once, hence the total
          running time is  O(n · number_of_divisors)  ≲ 1.3·10^7 operations,
          well inside the limits.
        • We only store a few small arrays, so the memory usage is  O(1).
        """
        n = len(s)
        
        # collect all divisors of n (ascending)
        divisors = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divisors.append(i)
                if i * i != n:
                    divisors.append(n // i)
            i += 1
        divisors.sort()                       # we need the minimum one
        
        # helper: build letter–frequency array of a substring [l, r) of s
        # (26 small ints, much faster than Counter)
        def freq(l: int, r: int) -> list:
            arr = [0] * 26
            for k in range(l, r):
                arr[ord(s[k]) - 97] += 1
            return arr
        
        # try the divisors from small to large
        for L in divisors:
            ref = freq(0, L)                  # reference block
            ok = True
            for start in range(L, n, L):
                if freq(start, start + L) != ref:
                    ok = False
                    break
            if ok:
                return L
        
        return n      # The entire string is always a trivial answer