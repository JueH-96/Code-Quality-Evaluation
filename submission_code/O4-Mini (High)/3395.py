class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        # gather all divisors of n
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        divs.sort()
        
        # we'll reuse a 26‐length list for counting
        base_a = ord('a')
        for L in divs:
            k = n // L
            # count of the first block
            base_count = [0] * 26
            for ch in s[:L]:
                base_count[ord(ch) - base_a] += 1
            
            ok = True
            block_count = [0] * 26
            # check all subsequent blocks
            for b in range(1, k):
                # reset block_count to zeros
                for i in range(26):
                    block_count[i] = 0
                start = b * L
                end = start + L
                for ch in s[start:end]:
                    block_count[ord(ch) - base_a] += 1
                if block_count != base_count:
                    ok = False
                    break
            
            if ok:
                return L
        
        # fallback (shouldn't be needed since L = n always works)
        return n