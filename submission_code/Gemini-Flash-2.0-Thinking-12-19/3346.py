class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        s_list = list(s)
        n = len(s)
        for i in range(n):
            original_char = s_list[i]
            for char_code in range(ord('a'), ord(original_char) + 1):
                char = chr(char_code)
                val1 = ord(original_char) - ord('a')
                val2 = ord(char) - ord('a')
                diff = abs(val1 - val2)
                distance = min(diff, 26 - diff)
                if distance <= k:
                    s_list[i] = char
                    k -= distance
                    break
        return "".join(s_list)

    def get_distance(self, s1_char, s2_char):
        val1 = ord(s1_char) - ord('a')
        val2 = ord(s2_char) - ord('a')
        diff = abs(val1 - val2)
        return min(diff, 26 - diff)