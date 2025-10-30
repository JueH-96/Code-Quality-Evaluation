class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        # initial count of ones in s
        initial_ones = s.count('1')
        
        # Build prefix sum for ones: pre[i] = number of ones in s[0:i]
        pre = [0]*(n+1)
        for i in range(n):
            pre[i+1] = pre[i] + (1 if s[i]=='1' else 0)
            
        # best gain from any valid trade substring.
        # For any interval [L, R] (0-indexed) that satisfies the "augmented border" conditions:
        #    if L > 0, then s[L-1] must be '0'
        #    if R < n-1, then s[R+1] must be '0'
        #
        # Then if we were allowed to “trade” on the substring s[L:R+1],
        # we can remove some ones and later flip a block of zeros so that the entire interval becomes ones.
        # This trade “improves” the ones count by:
        #      improvement = (length of interval) - (number of ones originally in the interval)
        # (Because after trade the entire interval becomes ones.)
        # Notice that if the trade could convert the whole interval then the final count would be
        #      initial_ones + improvement, yet we cannot exceed n.
        #
        # We search over all intervals that satisfy the border conditions.
        best = 0
        # We will “slide” over all intervals in O(n): we fix L and then try increasing R.
        # In order to get an O(n) solution, we observe that when scanning,
        # it is always optimal to take as long an interval as possible until the border condition fails.
        #
        # The idea is: 
        #   (1) An interval [L,R] is valid if (L == 0 or s[L-1]=='0') and (R == n-1 or s[R+1]=='0').
        #   (2) For each possible starting index L that satisfies the left border condition,
        #       extend R as far as possible (so that eventually the right border will be valid).
        #   (3) Among those, compute improvement = (R-L+1) - (ones in [L,R]).
        #
        # We can scan over L and, using a pointer R that always moves forward, try to maintain that
        # R is the farthest index such that R is either n-1 or s[R+1]=='0'. (For L, we only start if L==0 or s[L-1]=='0'.)
        #
        # (There are several equally-correct methods. In the worst case the scanning is O(n^2) but since each position
        # is only “entered” when it is a border candidate, and often the zeros are sparse, the average is linear.)
        
        # Here we use a two-loop approach.
        for L in range(n):
            # check left border condition: if L==0 then ok; otherwise the character before must be '0'
            if L > 0 and s[L-1] != '0':
                continue
            # now, try intervals starting at L
            # To satisfy the right border condition, for an interval ending at R < n-1, we require s[R+1]=='0'
            # or if R == n-1 it is automatically valid.
            for R in range(L, n):
                if R < n-1 and s[R+1] != '0':
                    continue
                # interval s[L:R+1] is valid
                length = R - L + 1
                ones_count = pre[R+1] - pre[L]
                improvement = length - ones_count
                if improvement > best:
                    best = improvement
        # the trade is optional, and also we cannot exceed n.
        return min(n, initial_ones + best)