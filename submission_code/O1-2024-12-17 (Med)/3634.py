class Solution:
    def calculateScore(self, s: str) -> int:
        # Precompute the mirror for each lowercase letter
        # mirror('a') = 'z', mirror('b') = 'y', etc.
        # This can be done with chr and ord.
        # mirror_map[c] gives the mirror character of c.
        mirror_map = {}
        for i in range(26):
            c1 = chr(ord('a') + i)
            c2 = chr(ord('z') - i)
            mirror_map[c1] = c2

        # pos[c] will hold the indices where character c occurs and is still unmarked.
        # We'll store them in ascending order as we go.
        pos = [[] for _ in range(26)]  # 26 lists, one for each letter 'a'..'z'
        
        score = 0
        
        for i, char in enumerate(s):
            # Find mirror char
            mirror_char = mirror_map[char]
            
            # Convert both characters to index 0..25
            mirror_idx = ord(mirror_char) - ord('a')
            char_idx = ord(char) - ord('a')
            
            # Check if there's an unmarked index j with s[j] == mirror_char
            # The "closest" one is the largest j < i, stored at the end of the list.
            if pos[mirror_idx]:
                j = pos[mirror_idx].pop()
                score += (i - j)
            else:
                # No match found, so store this index for possible later matches
                pos[char_idx].append(i)
        
        return score