class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        best = ""
        # We want to choose a valid piece from some split.
        # Any valid piece is a contiguous substring word[i:j] where i and j come from a set of boundaries
        # forming a partition of word into numFriends parts.
        #
        # Notice that the boundaries must satisfy:
        #   positions[0] = 0, positions[-1] = n, and we choose numFriends-1 boundaries in between.
        # Thus, each piece is word[i:j] where:
        #   - Either i == 0 (first piece) or i is one of the internal boundaries.
        #   - Either j == n (last piece) or j is one of the internal boundaries.
        #
        # We now determine, for each starting index i (a potential boundary), the maximum j we can choose
        # so that word[i:j] can be one piece in a valid split.
        #
        # For a piece starting at index i:
        #   • If i >= numFriends-1:
        #         We can split the left part (word[0:i]) into at least numFriends-1 pieces, so any j from i+1 up to n works.
        #         In such a case, to maximize lexicographic order (where longer strings are larger if one is a prefix),
        #         choose j = n.
        #
        #   • If i < numFriends-1:
        #         The left part (word[0:i]) can only account for at most i pieces (each non empty).
        #         Therefore, we must reserve at least (numFriends-1 - i) characters for the right pieces.
        #         This means j can be at most: n - (numFriends-1 - i).
        #
        # For each starting index i, choose the substring word[i:j] with j determined as above.
        # The lexicographically largest among these candidates is our answer.
        
        for i in range(n):
            if i < numFriends - 1:
                # Ensure that the remaining part (word[j:n]) can be split into (numFriends-1 - i) pieces.
                j = n - (numFriends - 1 - i)
            else:
                j = n
            candidate = word[i:j]
            if candidate > best:
                best = candidate
        return best