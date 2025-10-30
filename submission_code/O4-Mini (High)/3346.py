class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        """
        Build the lexicographically smallest string t such that the cyclic distance
        between s and t is <= k. Greedily choose for each position the smallest
        character c from 'a' to 'z' whose distance from s[i] does not exceed the
        remaining budget k.
        """
        res = []
        k_rem = k
        for ch in s:
            orig_code = ord(ch)
            # Try each candidate character from 'a' to 'z'
            for code in range(ord('a'), ord('z') + 1):
                diff = abs(orig_code - code)
                cost = min(diff, 26 - diff)
                if cost <= k_rem:
                    # Accept this character
                    res.append(chr(code))
                    k_rem -= cost
                    break
        return "".join(res)