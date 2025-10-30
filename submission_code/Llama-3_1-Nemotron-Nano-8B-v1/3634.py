class Solution:
    def calculateScore(self, s: str) -> int:
        from collections import defaultdict

        # Precompute the mirror for each character and their indices
        char_indices = defaultdict(list)
        for idx, c in enumerate(s):
            char_indices[c].append(idx)
        
        # Initialize pointers for each character's indices list
        pointers = {}
        for c in char_indices:
            pointers[c] = len(char_indices[c]) - 1  # Start from the end of the list
        
        marked = [False] * len(s)
        score = 0
        n = len(s)
        
        for i in range(n):
            if marked[i]:
                continue  # Skip if already marked
            
            # Calculate the mirror character of s[i]
            current_char = s[i]
            mirror_char = chr(25 - (ord(current_char) - ord('a')))
            
            if mirror_char not in char_indices:
                continue  # No such character exists
            
            indices_list = char_indices[mirror_char]
            ptr = pointers.get(mirror_char, -1)
            
            # Find the largest j < i that is unmarked
            while ptr >= 0:
                j = indices_list[ptr]
                if j < i and not marked[j]:
                    break
                ptr -= 1
            else:
                continue  # No valid j found
            
            # Update the score and mark both indices
            score += i - j
            marked[i] = True
            marked[j] = True
            
            # Move the pointer to the next possible position
            pointers[mirror_char] = ptr - 1
        
        return score