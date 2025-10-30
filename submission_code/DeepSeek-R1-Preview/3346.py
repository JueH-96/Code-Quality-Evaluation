class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        s_list = list(s)
        for i in range(len(s_list)):
            current_char = s_list[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c == current_char:
                    distance = 0
                else:
                    diff = abs(ord(c) - ord(current_char))
                    distance = min(diff, 26 - diff)
                if distance <= k:
                    s_list[i] = c
                    k -= distance
                    break
        return ''.join(s_list)