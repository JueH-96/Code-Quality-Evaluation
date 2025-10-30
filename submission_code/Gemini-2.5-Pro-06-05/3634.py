import collections

class Solution:
    def calculateScore(self, s: str) -> int:
        
        # This dictionary will store lists of indices for characters that are
        # currently "unmarked" and waiting for a mirror pair.
        # Each list acts as a stack (LIFO) to ensure we always find the
        # most recent (and thus closest) available index `j`.
        unmatched_stacks = collections.defaultdict(list)
        
        total_score = 0
        
        # The ordinal value of a character plus its mirror is a constant.
        # e.g., ord('a') + ord('z') = 97 + 122 = 219.
        # This allows for a quick calculation of the mirror character.
        mirror_ord_sum = ord('a') + ord('z')

        for i, char in enumerate(s):
            mirror_char = chr(mirror_ord_sum - ord(char))
            
            # Check if a stack exists for the mirror character and is not empty.
            if unmatched_stacks[mirror_char]:
                # A match is found. The index at the top of the mirror character's stack
                # is the most recently seen, which corresponds to the "closest" unmarked
                # index `j < i`.
                j = unmatched_stacks[mirror_char].pop()
                
                # By pairing them, we "mark" indices `i` and `j`. `j` is removed from the
                # stack, and `i` is not added to any stack.
                # Add the score difference to the total.
                total_score += i - j
            else:
                # No match found. The current character at index `i` is now an
                # "unmarked" candidate for subsequent characters. We push its index
                # onto its own character's stack.
                unmatched_stacks[char].append(i)
                
        return total_score