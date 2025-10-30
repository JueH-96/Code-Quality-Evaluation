class Solution:
    def calculateScore(self, s: str) -> int:
        # To find the mirror of each character, we can use the relationship:
        # mirror(c) = chr(ord('z') - (ord(c) - ord('a')))
        
        # We will maintain a stack (list) of indices for each character 'a' to 'z'.
        # As we iterate over s from left to right (index i):
        #   1) Compute mirror_c, the mirror character of s[i].
        #   2) Check if we have any unmarked indices stored for mirror_c. If so,
        #      pop the last index j (which is automatically the closest available)
        #      and add (i - j) to the total score. We mark i and j as used by
        #      removing j from the stack and not adding i to any stack.
        #   3) If there's no available j for mirror_c, push i onto the stack for s[i].
        
        # This works in O(n) time, because each index is pushed/popped exactly once.
        
        # Create 26 stacks, one for each lowercase letter:
        # index 0 -> 'a', 1 -> 'b', ..., 25 -> 'z'
        stacks = [[] for _ in range(26)]
        score = 0
        
        for i, ch in enumerate(s):
            # Compute the mirror character
            mirror_c_ord = ord('z') - (ord(ch) - ord('a'))
            # Convert mirror_c to an index 0..25
            mirror_c_idx = mirror_c_ord - ord('a')
            
            # If there's an available index in that mirror stack, pop it
            if stacks[mirror_c_idx]:
                j = stacks[mirror_c_idx].pop()  # closest unmarked index
                score += (i - j)
            else:
                # Otherwise, push this index onto the stack for ch
                ch_idx = ord(ch) - ord('a')
                stacks[ch_idx].append(i)
        
        return score