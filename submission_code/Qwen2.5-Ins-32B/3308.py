class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Count the frequency of each character in the string
        char_count = {}
        for char in s:
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
        
        # Find the maximum frequency
        max_freq = max(char_count.values())
        
        # Collect characters that have the maximum frequency
        max_freq_chars = [char for char, count in char_count.items() if count == max_freq]
        
        # Traverse the string from the end to find the last occurrence of these characters
        result = []
        for char in reversed(s):
            if char in max_freq_chars:
                result.append(char)
                max_freq_chars.remove(char)
        
        # Reverse the result to maintain the original order
        return ''.join(reversed(result))