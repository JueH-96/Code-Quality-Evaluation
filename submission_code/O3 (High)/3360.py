class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        """
        Let the frequency of every letter in the current (remaining) word lie in an
        interval [m , m+k].  For a fixed m   ( 1 ≤ m ≤ max_frequency ) the cheapest
        way to obtain such an interval is

            – for a letter whose original frequency f < m : delete the whole letter
              ( cost  = f )
            – for a letter with m ≤ f ≤ m+k          : keep it unchanged
            – for a letter with         f  > m+k     : trim it down to m+k
              ( cost = f-(m+k) )

        The total deletions for that m are therefore

            Σ(  f                       , if f < m
              | f-(m+k)                 , if f > m+k
              | 0                       , otherwise )

        We simply try every possible m ( at most 10^5 but for only 26 letters,
        so about 2.6 million primitive operations – easily fast enough ) and keep
        the minimum cost encountered.
        """
        # frequencies of the 26 lower–case letters
        freq = [0]*26
        for ch in word:
            freq[ord(ch)-97] += 1
        
        max_f = max(freq)
        if k >= max_f:               # the whole word is already k-special
            return 0
        
        best = len(word)             # upper-bound: delete everything
        
        for m in range(1, max_f+1):  # candidate for the minimum kept frequency
            cost = 0
            for f in freq:
                if f < m:                 # delete the letter completely
                    cost += f
                elif f > m + k:            # trim it down to m+k
                    cost += f - (m + k)
                # else: f in [m, m+k]  →  no deletion for this letter
                if cost >= best:           # small pruning
                    break
            best = min(best, cost)
        
        return best