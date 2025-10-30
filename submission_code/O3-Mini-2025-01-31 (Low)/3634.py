class Solution:
    def calculateScore(self, s: str) -> int:
        # Precompute mirror for each letter: mirror of letter = letter from reversed alphabet.
        # 'a' (ascii 97) to 'z' (ascii 122), so mirror of char c is:
        # chr(122 - (ord(c) - 97))
        def mirror_char(c: str) -> str:
            return chr(122 - (ord(c) - 97))
        
        # Use a dictionary to store stacks for unmarked indices for each letter.
        letter_stacks = {}
        total_score = 0
        
        # Iterate through the string:
        for i, char in enumerate(s):
            # What letter do we need to match?
            mirror = mirror_char(char)
            # Check if there's an available unmarked index for the mirror letter.
            if mirror in letter_stacks and letter_stacks[mirror]:
                # Pop the closest unmarked index j
                j = letter_stacks[mirror].pop()
                total_score += i - j
            else:
                # No matching index found, push the current index for potential future matches.
                if char not in letter_stacks:
                    letter_stacks[char] = []
                letter_stacks[char].append(i)
                
        return total_score