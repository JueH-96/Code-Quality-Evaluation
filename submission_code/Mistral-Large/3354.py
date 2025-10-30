class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Initialize a list to keep track of character counts
        char_count = [0] * 26
        # Initialize a list to store the result
        result = []

        for char in s:
            if char == '?':
                # Find the least frequent character to minimize the cost
                min_char = min(range(26), key=lambda i: (char_count[i], i))
                result.append(chr(ord('a') + min_char))
                char_count[min_char] += 1
            else:
                # Add the character to the result and update the count
                result.append(char)
                char_count[ord(char) - ord('a')] += 1

        return ''.join(result)