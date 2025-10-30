class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        res = []
        rem_k = k
        for char in s:
            pos_s = ord(char) - ord('a')
            if min(pos_s, 26 - pos_s) <= rem_k:
                cost = min(pos_s, 26 - pos_s)
                new_char = 'a'
                rem_k -= cost
            else:
                p_min = max(0, pos_s - rem_k)
                new_char = chr(p_min + ord('a'))
                rem_k = 0  # Use up all remaining k for this change
            res.append(new_char)
        return ''.join(res)