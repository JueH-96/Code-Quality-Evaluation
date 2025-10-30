class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        if numFriends == 1:
            return word
        if numFriends == n:
            return max(word)
        
        lcp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if word[i] == word[j]:
                    lcp[i][j] = lcp[i + 1][j + 1] + 1
                else:
                    lcp[i][j] = 0
        
        max_char = max(word)
        max_len = n - numFriends + 1
        candidate = None
        
        for i in range(n):
            if word[i] != max_char:
                continue
            best_j = None
            end = min(i + max_len, n)
            for j in range(i, end):
                left_len = i
                right_len = n - j - 1
                x_low = max(0, (numFriends - 1) - right_len)
                if left_len > 0:
                    x_low = max(x_low, 1)
                x_high = min(left_len, numFriends - 1)
                if right_len > 0:
                    x_high = min(x_high, numFriends - 2)
                if x_low <= x_high:
                    best_j = j
            if best_j is None:
                continue
            if candidate is None:
                candidate = (i, best_j)
            else:
                i0, j0 = candidate
                i1, j1 = i, best_j
                common = lcp[i0][i1]
                len0 = j0 - i0 + 1
                len1 = j1 - i1 + 1
                common_len = min(common, len0, len1)
                if common_len < len0 and common_len < len1:
                    if word[i0 + common_len] < word[i1 + common_len]:
                        candidate = (i1, j1)
                else:
                    if len0 < len1:
                        candidate = (i1, j1)
        return word[candidate[0]:candidate[1] + 1]