class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        k = numFriends
        # if there is only one friend, the only segment is the entire string.
        if k == 1:
            return word

        # We'll collect our candidate segments from each group.
        candidates = []
        
        # Group 1: first segment is forced to start at 0.
        # The first cut can be as far right as n - (k - 1)
        end1 = n - (k - 1)
        cand1 = word[0:end1]
        candidates.append(cand1)
        
        # For groups 2 to k-1 (middle segments)
        # For the j-th segment (with j = 2,...,k-1), the boundary for the previous segment (b_{j-1})
        # can be chosen in the range [L, U] where:
        #   L = j - 1   and   U = n - k + j - 1.
        # We then take the j-th segment to extend as far right as possible,
        # i.e. we fix its end at E = n - (k - j).
        # Among all allowed starting positions, we choose the one that makes s[i:E] lex–largest.
        for j in range(2, k):
            L = j - 1
            U = n - k + j - 1   # inclusive; note: U - L + 1 is a constant = n - k + 1.
            end_j = n - (k - j)
            best_index = L
            best_char = word[L]
            # In comparing substrings that all end at the same index end_j,
            # it suffices to take the i with maximum letter (and if tied, the smallest i wins).
            for i in range(L + 1, U + 1):
                ch = word[i]
                if ch > best_char:
                    best_char = ch
                    best_index = i
            cand = word[best_index: end_j]
            candidates.append(cand)

        # Group k: the last segment always ends at n.
        # b_{k-1} can be chosen from [k - 1, n - 1].
        L = k - 1
        U = n - 1
        end_k = n
        best_index = L
        best_char = word[L]
        for i in range(L + 1, U + 1):
            ch = word[i]
            if ch > best_char:
                best_char = ch
                best_index = i
        cand_k = word[best_index: end_k]
        candidates.append(cand_k)
        
        # The answer is the lexicographically maximum candidate.
        # (Python's built‐in max on strings works lexicographically.)
        return max(candidates)