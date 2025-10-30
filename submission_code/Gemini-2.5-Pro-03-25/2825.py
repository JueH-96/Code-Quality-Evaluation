import collections # This import is not strictly necessary as set is a built-in type.

class Solution:
    def minimizedStringLength(self, s: str) -> int:
        """
        Calculates the minimum length of the string after applying the described operation any number of times.

        The operation is defined as:
        Choose an index i in the string, and let c be the character in position i. 
        Delete the closest occurrence of c to the left of i (if any) and the closest occurrence of c to the right of i (if any).

        Analysis of the operation:
        Consider a character `c` that appears multiple times in the string `s`. Let its indices be `i_1 < i_2 < ... < i_k`.
        - If we choose an index `i_m` where `1 < m < k`, the operation will delete the character at index `i_{m-1}` (closest left `c`) and the character at index `i_{m+1}` (closest right `c`). The character `s[i_m]` remains. This reduces the count of `c` by 2.
        - If we choose index `i_1`, it deletes `s[i_2]` (closest right `c`). The count reduces by 1.
        - If we choose index `i_k`, it deletes `s[i_{k-1}]` (closest left `c`). The count reduces by 1.

        Through repeated applications of this operation, we can eliminate all duplicate occurrences of any character `c`. For instance, if `s = "aaaa"`:
        1. Choose `i=1` (`c='a'`). Delete `s[0]` and `s[2]`. String becomes `"aa"` (corresponding to original indices 1 and 3).
        2. In `"aa"`, choose `i=0` (`c='a'`). Delete `s[1]`. String becomes `"a"`.
        The count of 'a' is reduced from 4 to 1.

        This reduction process can be applied independently to each character type present in the string. An operation centered at a character `c` only deletes other instances of `c`. It does not delete instances of other characters.
        
        Furthermore, if a character `c` exists in the original string, at least one instance of `c` must remain in the final minimized string. This is because `c` can only be deleted by an operation centered at `c`. We can always perform the operations in such a way that one instance of `c` is preserved.

        Therefore, the minimum possible length of the string after applying the operation any number of times is equal to the number of unique characters present in the original string `s`.

        To find this minimum length, we can count the number of unique characters in `s`.

        Args:
            s: The input string, 0-indexed, containing lowercase English letters.

        Returns:
            An integer denoting the minimum length of the minimized string.
        """
        
        # Create a set from the characters of the string `s`.
        # A set automatically stores only unique elements.
        unique_characters = set(s)
        
        # The length of the set gives the count of unique characters.
        # This count represents the minimum length of the string achievable.
        min_length = len(unique_characters)
        
        return min_length