class Solution:
    def reverseDegree(self, s: str) -> int:
        # Pre-calculate the ASCII value of 'a'. This is used to efficiently
        # determine the 0-indexed alphabetical position of any lowercase letter
        # (e.g., 'a' maps to 0, 'b' to 1, ..., 'z' to 25).
        ord_a = ord('a')
        
        # The reverse degree is calculated by summing products for each character.
        # The sum() function is used with a generator expression for a concise solution.
        # For each character `char` at 0-indexed position `i` in the string `s`:
        # 1. `(ord(char) - ord_a)` calculates its 0-indexed alphabetical position.
        # 2. `26 - (ord(char) - ord_a)` converts this to the character's value
        #    in the reversed alphabet (where 'a'=26, 'b'=25, ..., 'z'=1).
        #    Example for 'a': 26 - (ord('a') - ord_a) = 26 - 0 = 26.
        #    Example for 'z': 26 - (ord('z') - ord_a) = 26 - 25 = 1.
        # 3. `(i + 1)` gives the 1-indexed position of the character in the string.
        # These two values (reversed alphabet position and string position) are multiplied.
        # The sum() function then adds up these products for all characters in `s`.
        
        calculated_reverse_degree = sum(
            (26 - (ord(char) - ord_a)) * (i + 1)
            for i, char in enumerate(s)
        )
        
        return calculated_reverse_degree