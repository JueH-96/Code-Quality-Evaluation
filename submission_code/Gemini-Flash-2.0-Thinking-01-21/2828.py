class Solution:
    def smallestString(self, s: str) -> str:
        chars = list(s)
        start_index = -1
        for i in range(len(chars)):
            if chars[i] != 'a':
                start_index = i
                break
        if start_index == -1:
            chars[-1] = 'z'
        else:
            for j in range(start_index, len(chars)):
                if s[j] == 'a':
                    break
                else:
                    chars[j] = chr(ord(chars[j]) - 1)
        return "".join(chars)