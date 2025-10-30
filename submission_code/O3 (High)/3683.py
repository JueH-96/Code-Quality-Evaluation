class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        k = numFriends
        
        # With one friend the whole word is the only possible piece.
        if k == 1:
            return word
        
        best = ""
        k_minus_1 = k - 1
        
        for i in range(n):
            if i >= k_minus_1:
                # we can place all the remaining (k-1) segments before the current
                # position, so the segment may stretch to the end of the word
                candidate = word[i:]
            else:
                # we must leave (k-1-i) characters to the right of the segment
                # so that they can form the remaining segments (one per character)
                j_max = n - (k_minus_1 - i)          # last index (exclusive)
                candidate = word[i:j_max]            # longest feasible segment
            if candidate > best:
                best = candidate
        return best