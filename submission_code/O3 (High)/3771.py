class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        """
        Return True if we can choose k pairwise disjoint special substrings
        from s, otherwise return False.
        A special substring contains only characters that do NOT occur
        outside that substring and the substring must not be the whole string.
        """
        n = len(s)
        # k == 0  ->  always possible
        if k == 0:
            return True
        # With only 26 letters we can never obtain more than 26 disjoint
        # special substrings.
        if k > 26:
            return False

        # 1. first and last position of every letter
        first = [n] * 26
        last  = [-1] * 26
        for i, ch in enumerate(s):
            idx = ord(ch) - 97
            if first[idx] == n:
                first[idx] = i
            last[idx] = i

        # 2. build every minimal valid interval that can possibly start
        #    at a position which is the first appearance of its character.
        intervals = []          # we store them as (end, start) pairs
        for c in range(26):
            if first[c] == n:           # letter not present
                continue
            l = first[c]
            end = last[c]
            valid = True
            j = l
            while j <= end:
                cur = ord(s[j]) - 97
                # if this letter has an occurrence before l,
                # then the substring would contain that letter
                # both inside and outside -> invalid
                if first[cur] < l:
                    valid = False
                    break
                end = max(end, last[cur])
                j += 1
            # ignore the whole string itself
            if valid and (end - l + 1) != n:
                intervals.append((end, l))

        # 3. choose the maximum number of non-overlapping intervals
        intervals.sort()                 # sort by ending position
        taken = 0
        prev_end = -1
        for end, start in intervals:
            if start > prev_end:         # disjoint
                taken += 1
                prev_end = end
                if taken >= k:           # early exit
                    return True

        return False