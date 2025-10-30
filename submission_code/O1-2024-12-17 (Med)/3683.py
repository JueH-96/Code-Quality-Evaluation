class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        """
        We want the lexicographically largest substring that can appear as one of the
        numFriends consecutive (non-empty) pieces in a partition of 'word'.

        It can be shown that a substring word[i : i+length] is a valid piece if and only if
        1 <= length <= (len(word) - numFriends + 1)
        and there is some way to place it in the k-th slot (1 <= k <= numFriends)
        so that all numFriends parts are non-empty and cover the string exactly.
        
        In fact, the condition for feasibility simplifies to:
           length <= (len(word) - numFriends + 1).
        
        Hence, for each starting position i, the largest feasible piece is
           word[i : i + min( (len(word) - i), (len(word) - numFriends + 1) )].
        
        We pick the lexicographically largest among these candidates.
        """

        n = len(word)
        # Maximum length any single piece can have
        max_piece_len = n - numFriends + 1  # This is always >= 1 by problem constraints

        best_substring = ""
        for i in range(n):
            # The longest valid piece starting at i
            length = min(n - i, max_piece_len)
            if length <= 0:
                # No more valid pieces from here onward
                break
            candidate = word[i : i + length]
            if candidate > best_substring:
                best_substring = candidate
        
        return best_substring