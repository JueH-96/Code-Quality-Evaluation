class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        min_length = float('inf')
        smallest_substring = ""
        
        # Iterate through all possible substrings
        for start in range(len(s)):
            for end in range(start, len(s)):
                substring = s[start:end+1]
                count_ones = substring.count('1')
                if count_ones == k:
                    length = end - start + 1
                    if length < min_length:
                        min_length = length
                        smallest_substring = substring
                    elif length == min_length:
                        if substring < smallest_substring:
                            smallest_substring = substring
        return smallest_substring