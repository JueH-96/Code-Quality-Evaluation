class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        t_list = list(s)

        def char_distance(c1, c2):
            ord1 = ord(c1) - ord('a')
            ord2 = ord(c2) - ord('a')
            diff = abs(ord1 - ord2)
            return min(diff, 26 - diff)

        for i in range(n):
            original_char = s[i]
            for char_code in range(ord('a'), ord('z') + 1):
                candidate_char = chr(char_code)
                dist = char_distance(original_char, candidate_char)

                if k >= dist:
                    k -= dist
                    t_list[i] = candidate_char
                    break

        return "".join(t_list)