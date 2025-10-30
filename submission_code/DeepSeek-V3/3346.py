class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        t = list(s)
        n = len(t)
        for i in range(n):
            current_char = t[i]
            for target_char in 'abcdefghijklmnopqrstuvwxyz':
                if target_char >= current_char:
                    distance = ord(target_char) - ord(current_char)
                else:
                    distance = 26 - (ord(current_char) - ord(target_char))
                if distance <= k:
                    t[i] = target_char
                    k -= distance
                    break
        return ''.join(t)