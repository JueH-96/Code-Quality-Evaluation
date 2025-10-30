class Solution:
    def reverseDegree(self, s: str) -> int:
        total_reverse_degree = 0
        
        # 'z' has ASCII value 122, 'a' has 97.
        # The position in the reversed alphabet for 'a' is 26, for 'z' is 1.
        # This can be calculated as: (ASCII of 'z' - ASCII of char) + 1
        # Example: for 'a': (122 - 97) + 1 = 25 + 1 = 26
        # Example: for 'z': (122 - 122) + 1 = 0 + 1 = 1

        for i, char_val in enumerate(s):
            # Calculate the character's position in the reversed alphabet
            # ord('z') - ord(char_val) gives the 0-indexed distance from 'z'
            # Adding 1 makes it 1-indexed as per the problem ('z' = 1)
            reversed_alphabet_pos = ord('z') - ord(char_val) + 1
            
            # Calculate the character's 1-indexed position in the string
            # enumerate gives a 0-indexed 'i', so add 1 for 1-indexed position
            string_pos = i + 1
            
            # Multiply these two positions and add to the total
            total_reverse_degree += (reversed_alphabet_pos * string_pos)
            
        return total_reverse_degree