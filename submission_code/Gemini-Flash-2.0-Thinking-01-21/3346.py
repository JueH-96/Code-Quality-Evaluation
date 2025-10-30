class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        s_list = list(s)
        n = len(s)
        for i in range(n):
            original_char = s_list[i]
            for char_code in range(ord('a'), ord('z') + 1):
                test_char = chr(char_code)
                diff = abs(ord(original_char) - ord(test_char))
                dist = min(diff, 26 - diff)
                if dist <= k:
                    s_list[i] = test_char
                    k -= dist
                    break
        return "".join(s_list)

def char_distance(c1, c2):
    diff = abs(ord(c1) - ord(c2))
    return min(diff, 26 - diff)