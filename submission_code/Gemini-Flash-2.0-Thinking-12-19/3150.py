class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        shortest_len = float('inf')
        result = ""
        found_beautiful_substring = False

        for length in range(1, len(s) + 1):
            for start_index in range(len(s) - length + 1):
                sub = s[start_index:start_index + length]
                ones_count = sub.count('1')
                if ones_count == k:
                    found_beautiful_substring = True
                    if length < shortest_len:
                        shortest_len = length
                        result = sub
                    elif length == shortest_len:
                        if result == "" or sub < result:
                            result = sub

        if not found_beautiful_substring:
            return ""
        return result