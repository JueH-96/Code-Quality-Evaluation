class Solution:
    def minimizeStringValue(self, s: str) -> str:
        s_list = list(s)
        n = len(s_list)
        # First, count the frequency of each character in the fixed positions
        freq = {}
        for i in range(n):
            if s_list[i] != '?':
                freq[s_list[i]] = freq.get(s_list[i], 0) + 1
        # Now, for each '?', assign the character that minimizes the cost
        # We want to assign the character with the smallest current frequency
        # and in case of a tie, the lex smallest character
        for i in range(n):
            if s_list[i] == '?':
                # Find the character with the smallest frequency
                min_char = 'a'
                min_freq = freq.get(min_char, 0)
                for c in range(ord('a'), ord('z') + 1):
                    char = chr(c)
                    current_freq = freq.get(char, 0)
                    if current_freq < min_freq:
                        min_char = char
                        min_freq = current_freq
                # Assign the character
                s_list[i] = min_char
                # Update the frequency
                freq[min_char] = freq.get(min_char, 0) + 1
        return ''.join(s_list)