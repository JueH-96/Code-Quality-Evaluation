class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        # Count the frequency needed from word2
        from collections import Counter
        freq_word2 = Counter(word2)
        
        # "missing" will track how many total characters (across all needed frequencies)
        # are still needed to make the current window valid.
        # For example, if word2 = "aabc", then freq_word2['a'] = 2, freq_word2['b'] = 1,
        # freq_word2['c'] = 1, and missing = 4 initially.
        missing = sum(freq_word2.values())
        
        freq_window = Counter()
        left = 0
        result = 0
        n = len(word1)
        
        for right, ch in enumerate(word1):
            # Add current character to the window frequency
            freq_window[ch] += 1
            
            # If this character is still within the needed count, reduce "missing"
            # (i.e., we've fulfilled one more character requirement of word2)
            if freq_window[ch] <= freq_word2[ch]:
                missing -= 1
            
            # While we have a valid window (missing == 0), try shrinking from the left
            # Each time missing == 0, it means the substring from left..right
            # has at least as many of each character as word2 needs.
            while missing == 0:
                # Since left..right is valid, any substring extending from right to the end
                # still remains valid (adding extra characters can't break "≥" condition).
                # Therefore, we add (n - right) to the result.
                result += (n - right)
                
                # Now try to shrink the window from the left
                left_char = word1[left]
                freq_window[left_char] -= 1
                # If removing this character breaks the requirement,
                # we increase missing and stop shrinking further.
                if freq_window[left_char] < freq_word2[left_char]:
                    missing += 1
                left += 1
        
        return result