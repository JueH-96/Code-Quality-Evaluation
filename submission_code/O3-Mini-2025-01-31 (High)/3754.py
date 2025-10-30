class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)
        # Precompute prefix frequency counts for each letter.
        # For 0 <= i <= n, prefX[i] is the number of times letter X appears in s[0:i].
        prefN = [0] * (n + 1)
        prefS = [0] * (n + 1)
        prefE = [0] * (n + 1)
        prefW = [0] * (n + 1)
        for i in range(n):
            prefN[i+1] = prefN[i] + (1 if s[i] == 'N' else 0)
            prefS[i+1] = prefS[i] + (1 if s[i] == 'S' else 0)
            prefE[i+1] = prefE[i] + (1 if s[i] == 'E' else 0)
            prefW[i+1] = prefW[i] + (1 if s[i] == 'W' else 0)
        
        # Explained idea:
        # In an "ideal" path that always increases the Manhattan distance (|x|+|y|) by 1 each step,
        # every move must be “beneficial”. When starting at (0,0), an optimal beneficial
        # route is one where every move “pushes” the position farther from the origin.
        # Because moves are one unit, the maximum Manhattan distance achievable in L moves is L.
        #
        # It turns out that any such beneficial sequence (without back‐tracking) must use 
        # moves from two fixed groups: one horizontal move (always E or always W) and 
        # one vertical move (always N or always S). (In the degenerate case one “axis” might be unused.)
        # In other words, if we choose for example (X, Y) = (E, N) then a beneficial sequence is any interleaving
        # of E‐moves and N‐moves. In the resulting route the Manhattan distance after L moves is L.
        #
        # Now, we are allowed to change at most k characters in s. Think of “fixing” a prefix s[0:L]
        # so that every character is one of our target letters (one chosen from {E,W} for horizontal
        # and one chosen from {N,S} for vertical). For a fixed prefix we could choose any pair:
        # (E,N), (E,S), (W,N), or (W,S). If we decide to “force” a beneficial prefix of length L
        # using, say, (E,N), then we don’t need to change the characters that are already 'E' or 'N'
        # in the prefix; every other character must be modified.
        #
        # Thus, for a given L and a chosen pair (X, Y) (with X in {E, W} and Y in {N, S}),
        # the number of modifications required is:
        #     mods = L - (number of characters in s[0:L] that are either X or Y).
        #
        # Over the 4 possible choices of (X, Y), the best (lowest modifications needed) is:
        #     mods(L) = L - max( count(X) + count(Y) )  over (X,Y) in {(E,N),(E,S),(W,N),(W,S)}.
        #
        # Our goal is to find the maximum prefix length L (which equals the Manhattan distance achieved)
        # for which mods(L) <= k.
        
        ans = 0
        for L in range(n + 1):
            # Get counts for each letter in the prefix s[0:L]
            countN = prefN[L]
            countS = prefS[L]
            countE = prefE[L]
            countW = prefW[L]
            # For each beneficial pair we could try:
            # (E,N), (E,S), (W,N), (W,S)
            best_pair = max(countE + countN, countE + countS, countW + countN, countW + countS)
            modifications_needed = L - best_pair
            if modifications_needed <= k:
                ans = L
        return ans