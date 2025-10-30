class Solution:
    def clearDigits(self, s: str) -> str:
        s_list = list(s)
        while any(ch.isdigit() for ch in s_list):
            # Find index i of the first digit
            for idx, ch in enumerate(s_list):
                if ch.isdigit():
                    i = idx
                    break
            # Remove digit at index i
            s_list.pop(i)
            # Find the closest non-digit character to the left
            for j in range(i - 1, -1, -1):
                if not s_list[j].isdigit():
                    s_list.pop(j)
                    break
        return ''.join(s_list)