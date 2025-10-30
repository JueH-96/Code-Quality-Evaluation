class Solution:
    def calculateScore(self, s: str) -> int:
        score = 0
        
        def get_mirror(char):
            return chr(ord('z') - (ord(char) - ord('a')))
        
        # For each character, maintain a stack of unmarked indices
        char_indices = {}
        
        for i, char in enumerate(s):
            mirror_char = get_mirror(char)
            
            # Check if there's an unmarked index with the mirror character
            if mirror_char in char_indices and char_indices[mirror_char]:
                j = char_indices[mirror_char].pop()  # Get the closest (rightmost) index
                score += i - j
            else:
                # No match found, add this index to the list for this character
                if char not in char_indices:
                    char_indices[char] = []
                char_indices[char].append(i)
        
        return score