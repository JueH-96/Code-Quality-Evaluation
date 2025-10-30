import collections

class Solution:
  def calculateScore(self, s: str) -> int:
    
    # Helper function to find the mirror of a character.
    # e.g. mirror('a') = 'z', mirror('b') = 'y', ...
    def get_mirror_char(char_val: str) -> str:
        # The 0-indexed position of char_val (e.g., 'a' is 0, 'b' is 1).
        k = ord(char_val) - ord('a')
        # The 0-indexed position of its mirror (e.g., for 'a' (k=0), mirror_k is 25-0=25, which is 'z').
        mirror_k = 25 - k
        # Convert mirror_k back to a character.
        return chr(ord('a') + mirror_k)

    n = len(s)
    
    # char_idx_stacks will store lists of indices for each character.
    # These lists will function as stacks.
    # An index `p` is pushed to `char_idx_stacks[s[p]]` if, when `s[p]` was processed
    # (i.e., when current index `i` was `p`), `s[p]` did not find a match `s[j]` (j < p).
    # This means `s[p]` (and thus index `p`) is available ("unmarked") to be a match `s[j]`
    # for some future character `s[k]` (where `k > p`).
    # Being on a stack implies the character at that index is "unmarked".
    char_idx_stacks = collections.defaultdict(list)
    
    total_score = 0
    
    # Iterate through the string from left to right with index `i`.
    for i in range(n):
        current_char = s[i]
        mirror_of_current_char = get_mirror_char(current_char)
        
        # This flag tracks if s[i] forms a pair.
        # If it does, s[i] gets "marked" and is not available for future pairings as an s[j].
        # (It won't be pushed onto its character's stack).
        # If it doesn't, s[i] remains "unmarked" and is pushed onto its character's stack
        # to be potentially used as an s[j] later.
        current_char_found_match = False
        
        # Check if there's any available prior character that is the mirror of s[i].
        # char_idx_stacks[mirror_of_current_char] holds indices of such characters.
        # These indices are j < i, s[j] = mirror_of_current_char, and s[j] is unmarked.
        # The .pop() operation will retrieve the most recently added index,
        # which is the largest j (closest to i), satisfying the problem's criteria.
        if char_idx_stacks[mirror_of_current_char]:
            # An available mirror character s[j] exists. Pop its index j.
            j = char_idx_stacks[mirror_of_current_char].pop()
            
            # Add i-j to the score.
            total_score += (i - j)
            
            # s[i] and s[j] are now "marked".
            # s[i] is marked by setting current_char_found_match = True (so it won't be pushed to a stack).
            # s[j] is marked by being popped from the stack (so it can't be used again).
            current_char_found_match = True
        
        # If s[i] did not find a match:
        if not current_char_found_match:
            # s[i] is "unmarked" (for now) and available for future characters to match with it.
            # Push its index `i` onto the stack corresponding to `current_char`.
            char_idx_stacks[current_char].append(i)
            
    return total_score