class Solution:
    def getSmallestString(self, s: str) -> str:
        s = list(s)
        n = len(s)

        # Find the first pair of adjacent digits with the same parity that are not in increasing order
        for i in range(n - 1):
            if (int(s[i]) % 2 == int(s[i + 1]) % 2) and s[i] > s[i + 1]:
                # Swap them and return the result
                s[i], s[i + 1] = s[i + 1], s[i]
                return ''.join(s)

        # If no such pair is found, return the original string
        return ''.join(s)