class Solution:
    def reverseDegree(self, s: str) -> int:
        """
        Compute the reverse degree of the input string.

        For every character c at 1-based position i:
            reversed_alphabet_index = 26 - (ord(c) - ord('a'))
            contribution = reversed_alphabet_index * i
        Sum contributions for all characters.

        Parameters
        ----------
        s : str
            A non-empty string of lowercase English letters (1 ≤ len(s) ≤ 1000).

        Returns
        -------
        int
            The reverse degree of the string.
        """
        total = 0
        for i, ch in enumerate(s, start=1):       # 1-based index
            rev_idx = 26 - (ord(ch) - ord('a'))   # 'a'→26, 'b'→25 … 'z'→1
            total += rev_idx * i
        return total