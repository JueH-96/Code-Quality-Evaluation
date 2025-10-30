class Solution:
    def calculateScore(self, s: str) -> int:
        def get_mirror(c):
            return chr(ord('z') - (ord(c) - ord('a')))
        
        # Keep track of unmarked indices for each character
        char_indices = {}
        for i in range(len(s)):
            if s[i] not in char_indices:
                char_indices[s[i]] = []
            char_indices[s[i]].append(i)
        
        marked = set()
        score = 0
        
        for i in range(len(s)):
            if i in marked:
                continue
                
            mirror_char = get_mirror(s[i])
            
            # Find the closest unmarked index j < i with mirror character
            closest_j = -1
            
            if mirror_char in char_indices:
                for j in char_indices[mirror_char]:
                    if j < i and j not in marked:
                        closest_j = j
                    elif j >= i:
                        break
            
            if closest_j != -1:
                # Mark both indices and add to score
                marked.add(i)
                marked.add(closest_j)
                score += i - closest_j
        
        return score