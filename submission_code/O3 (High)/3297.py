class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        """
        Let n = len(word). After every second we delete the first k letters and
        append k new letters. 
        
        After t seconds the first (n - t*k) characters of the current string
        are exactly word[t*k:] (they are the part of the original word that
        has not been thrown away yet), while the last t*k characters are
        completely under our control (they were appended by us during the last
        t steps).

        Hence, to obtain the original word after t (>0) seconds two situations
        can make this possible:
            1) t*k >= n  – all original letters have been removed, so we can
               rebuild the whole word freely during those t steps;
            2) t*k  < n  – then the uncontrolled prefix that survives must
               already equal the corresponding prefix of the original word,
               i.e.  word[t*k:] == word[:n-t*k].

        The smallest t that satisfies one of the two conditions is the answer.
        Because n ≤ 50 we can just test t = 1,2,3,… until one of the
        conditions holds.
        """
        n = len(word)
        t = 1
        while True:                      # n is tiny, a simple loop suffices
            removed = t * k              # number of characters removed so far
            if removed >= n:             # case 1
                return t
            if word[removed:] == word[:n - removed]:   # case 2
                return t
            t += 1