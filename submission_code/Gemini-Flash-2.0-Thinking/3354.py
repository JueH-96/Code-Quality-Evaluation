class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        s_list = list(s)
        counts = {}
        for char_code in range(ord('a'), ord('z') + 1):
            counts[chr(char_code)] = 0

        for i in range(n):
            if s_list[i] == '?':
                min_count = float('inf')
                best_char = ''
                for char_code in range(ord('a'), ord('z') + 1):
                    char = chr(char_code)
                    if counts[char] < min_count:
                        min_count = counts[char]
                        best_char = char
                    elif counts[char] == min_count and char < best_char:
                        best_char = char
                s_list[i] = best_char
                counts[best_char] += 1
            else:
                counts[s_list[i]] += 1

        return "".join(s_list)