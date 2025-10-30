class Solution:
    def minimizeStringValue(self, s: str) -> str:
        # Convert string to list for easier manipulation
        s_list = list(s)
        
        # Track frequency of each character
        freq = [0] * 26
        
        # Count initial frequencies (non-'?' characters)
        for char in s:
            if char != '?':
                freq[ord(char) - ord('a')] += 1
        
        # For each '?' position, we need to determine which character to use
        # We'll collect all the characters we need to place
        chars_to_place = []
        temp_freq = freq[:]  # Copy of frequencies for simulation
        
        for char in s:
            if char == '?':
                # Find the character with minimum frequency
                min_freq = min(temp_freq)
                # Among characters with min frequency, choose the smallest
                for i in range(26):
                    if temp_freq[i] == min_freq:
                        chars_to_place.append(chr(i + ord('a')))
                        temp_freq[i] += 1
                        break
            else:
                # Just update the frequency for non-'?' characters
                pass
        
        # Sort the characters to place to ensure lexicographically smallest result
        chars_to_place.sort()
        
        # Place the sorted characters in the '?' positions
        char_idx = 0
        for i in range(len(s_list)):
            if s_list[i] == '?':
                s_list[i] = chars_to_place[char_idx]
                char_idx += 1
        
        return ''.join(s_list)