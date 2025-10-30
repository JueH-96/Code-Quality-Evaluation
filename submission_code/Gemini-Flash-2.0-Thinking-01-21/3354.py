class Solution:
    def minimizeStringValue(self, s: str) -> str:
        result_string = ""
        char_counts = {}
        for char_code in range(ord('a'), ord('z') + 1):
            char_counts[chr(char_code)] = 0
        
        for char_from_s in s:
            if char_from_s != '?':
                result_string += char_from_s
                char_counts[char_from_s] += 1
            else:
                min_count = float('inf')
                for char_code in range(ord('a'), ord('z') + 1):
                    char = chr(char_code)
                    min_count = min(min_count, char_counts[char])
                
                chosen_char = ''
                for char_code in range(ord('a'), ord('z') + 1):
                    char = chr(char_code)
                    if char_counts[char] == min_count:
                        chosen_char = char
                        break
                        
                result_string += chosen_char
                char_counts[chosen_char] += 1
                
        return result_string