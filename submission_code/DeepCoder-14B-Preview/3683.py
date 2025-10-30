class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        L = len(word)
        candidates = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                if i == 0:
                    if j == L:
                        continue  # Entire word, only possible if numFriends is 1, which is handled
                    else:
                        right_len = L - j
                        if right_len >= (numFriends - 1):
                            candidates.add(word[i:j])
                elif j == L:
                    left_len = i
                    if left_len >= (numFriends - 1):
                        candidates.add(word[i:j])
                else:
                    if (numFriends - 1 >= 2) and (i >= 1) and ((L - j) >= 1) and ((i + (L - j)) >= (numFriends - 1)):
                        candidates.add(word[i:j])
        if not candidates:
            return ""
        return max(candidates)