import collections

class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Step 1: Calculate frequency of each character
        freq = collections.Counter(s)
            
        # Step 2: Find the maximum frequency
        # Constraint 1 <= s.length guarantees s is not empty, so max(freq.values()) is safe.
        max_freq = max(freq.values())
        
        # Step 3 & 4: Iterate through s from right to left, collect characters
        # that have the maximum frequency and are encountered for the first time
        # during this right-to-left scan (meaning it's their last occurrence in s).
        result_chars_rev = []
        added_chars = set() # To track which max_freq chars we've added to result_chars_rev

        # Iterate backwards through the string indices
        for i in range(len(s) - 1, -1, -1):
            char = s[i]
            
            # If the current character's frequency is the maximum AND we haven't already
            # added this character to our result list (which implies this is the last
            # occurrence we see when scanning from the right), add it to our list.
            if freq[char] == max_freq and char not in added_chars:
                 result_chars_rev.append(char)
                 added_chars.add(char)

        # Step 5: The characters were collected from right to left, so they are
        # in reverse order of their last appearance in s. Reverse the list to get
        # the correct final string order.
        result_chars_rev.reverse()
        
        # Join the characters to form the result string.
        return "".join(result_chars_rev)