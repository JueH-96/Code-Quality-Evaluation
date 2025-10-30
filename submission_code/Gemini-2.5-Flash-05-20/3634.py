class Solution:
    def calculateScore(self, s: str) -> int:
        score = 0
        
        # unmarked_char_stacks[k] will store a list of indices j
        # where s[j] == chr(ord('a') + k) and j is currently unmarked.
        # We use a list as a stack (append for push, pop for pop).
        # Popping from the end of the list gives the largest (most recent) 
        # available index, which naturally fulfills the "closest" requirement
        # for j < i.
        unmarked_char_stacks = [[] for _ in range(26)] # 26 lowercase English letters

        for i in range(len(s)):
            # Convert current character s[i] to its 0-indexed position (0 for 'a', 1 for 'b', ..., 25 for 'z')
            current_char_idx = ord(s[i]) - ord('a')
            
            # Calculate the 0-indexed position of its mirror character.
            # 'a' (0) mirrors 'z' (25), 'b' (1) mirrors 'y' (24), etc.
            # The general formula is 25 - current_char_idx.
            mirror_char_idx = 25 - current_char_idx

            # Check if there's an unmarked index j for the mirror character
            if unmarked_char_stacks[mirror_char_idx]:
                # If the stack for the mirror character is not empty,
                # it means we found at least one unmarked index j to the left of i.
                
                # Pop the top element (the most recently added, hence closest)
                # from the stack of the mirror character.
                j = unmarked_char_stacks[mirror_char_idx].pop()
                
                # Add the distance i - j to the total score.
                score += (i - j)
                
                # Both indices i and j are now considered "marked" or "used".
                # j was removed from its stack.
                # i is not added to any stack, as it was immediately paired.
            else:
                # If no suitable j was found (the stack for the mirror character was empty),
                # then s[i] itself becomes an available unmarked character for future matches.
                # Add its index i to the stack corresponding to its own character.
                unmarked_char_stacks[current_char_idx].append(i)
                
        return score