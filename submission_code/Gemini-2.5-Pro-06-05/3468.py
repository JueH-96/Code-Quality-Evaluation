class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        """
        Encrypts the string s by cyclically shifting it.

        The encryption algorithm is equivalent to a left circular rotation
        of the string by k positions.

        Args:
            s: The input string of lowercase English letters.
            k: The integer shift amount.

        Returns:
            The encrypted string.
        """
        n = len(s)
        
        # The effective rotation amount is k % n, as rotations are cyclic.
        # For example, on a string of length 5, a rotation of 8 is the same as a rotation of 3.
        rotate_by = k % n
        
        # A left rotation by `rotate_by` positions can be achieved by
        # taking the suffix of the string starting at `rotate_by` and
        # prepending it to the prefix of length `rotate_by`.
        # Example: s = "dart", rotate_by = 3
        # s[3:] is "t"
        # s[:3] is "dar"
        # Result is "t" + "dar" = "tdar"
        return s[rotate_by:] + s[:rotate_by]