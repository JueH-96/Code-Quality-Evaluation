class Solution:
    def stringHash(self, s: str, k: int) -> str:
        """
        Hash a string by splitting it into chunks of size k, summing the 0-based
        alphabetical indices of each chunk, taking each sum modulo 26, and
        converting that value back to a character.

        Parameters
        ----------
        s : str
            Input string consisting of lowercase English letters.
        k : int
            Length of each chunk; guaranteed to divide len(s).

        Returns
        -------
        str
            The hashed string of length len(s) // k.
        """
        if k == 0 or not s:
            return ""

        n = len(s)
        # Precompute the numeric value (0‒25) of every character once
        vals = [ord(ch) - 97 for ch in s]

        res_chars = []
        for i in range(0, n, k):
            # Sum k values; Python slicing is fine given constraints
            chunk_sum = sum(vals[i:i + k])
            hashed_val = chunk_sum % 26
            res_chars.append(chr(hashed_val + 97))

        return ''.join(res_chars)