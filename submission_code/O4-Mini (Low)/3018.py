class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        """
        We are allowed a single operation: choose any subset of positions in str1
        and increment each chosen character to the next letter cyclically.
        After that operation, we want str2 to be a subsequence of the resulting str1.

        Observing that each character in str1 independently can either stay the same
        or be incremented once, we can treat each str1[i] as having two possible values:
        orig = str1[i] or inc = chr((ord(str1[i]) - ord('a') + 1) % 26 + ord('a'))

        We then greedily try to match str2 as a subsequence: scan str1 left to right
        with a pointer j into str2, and whenever str2[j] matches either orig or inc
        of the current str1 character, we advance j. If by the end j reaches len(str2),
        return True, else False.
        """
        n, m = len(str1), len(str2)
        j = 0
        for c in str1:
            if j < m:
                # original character
                if str2[j] == c:
                    j += 1
                else:
                    # incremented character
                    inc = chr((ord(c) - ord('a') + 1) % 26 + ord('a'))
                    if str2[j] == inc:
                        j += 1
            else:
                break
        return j == m