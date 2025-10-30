class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        """
        For every character in str1 we may either keep it as it is, or increase it
        once (cyclically, so 'z' -> 'a').  After this single mass-operation we want
        str2 to be a subsequence of the new str1.

        During matching each position of str1 therefore supplies a set of *two*
        possible characters:  {c, next(c)}.  We only need to decide whether str2
        can be matched as a subsequence against these sets – a standard two-pointer
        greedy scan suffices.
        """
        n, m = len(str1), len(str2)
        if m > n:          # subsequence can never be longer than the original
            return False

        i = j = 0          # i -> str1, j -> str2
        while i < n and j < m:
            c1 = str1[i]
            # character after c1 in the alphabet (wrap around after 'z')
            next_c1 = 'a' if c1 == 'z' else chr(ord(c1) + 1)

            if c1 == str2[j] or next_c1 == str2[j]:
                j += 1     # current char of str2 matched
            i += 1         # always advance in str1

        return j == m      # did we match all of str2?