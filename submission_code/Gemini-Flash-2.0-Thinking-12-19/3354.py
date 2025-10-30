class Solution:
    def minimizeStringValue(self, s: str) -> str:
        counts = {}
        for char_code in range(ord('a'), ord('z') + 1):
            counts[chr(char_code)] = 0
        result_string = []
        for char_s in s:
            if char_s == '?':
                min_count = float('inf')
                chosen_char = None
                for char_code in range(ord('a'), ord('z') + 1):
                    char = chr(char_code)
                    if counts[char] < min_count:
                        min_count = counts[char]
                        chosen_char = char
                    elif counts[char] == min_count:
                        if chosen_char is None or char < chosen_char:
                            chosen_char = char
                result_string.append(chosen_char)
                counts[chosen_char] += 1
            else:
                result_string.append(char_s)
                counts[char_s] += 1
        return "".join(result_string)