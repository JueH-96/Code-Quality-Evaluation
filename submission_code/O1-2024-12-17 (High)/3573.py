class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        # We want to count how many substrings of word1 can be rearranged
        # so that word2 is a prefix. In other words, each valid substring
        # must contain at least all the characters (with their frequencies)
        # that appear in word2.
        #
        # This boils down to counting all substrings of word1 whose character
        # frequencies are >= those in word2.
        #
        # We can do this efficiently with a two-pointer (sliding window) approach:
        # 1) Count the character frequencies of word2.
        # 2) Use two pointers (start, end) over word1. We expand 'end' until
        #    the substring word1[start:end+1] has at least freq2 of every character.
        #    The first time it becomes valid, every extension to the right
        #    remains valid, so we add (len(word1) - end) to the answer.
        # 3) Then move 'start' forward by one to consider the next starting position,
        #    removing its contribution from the window, and possibly moving 'end'
        #    further if needed. We keep track of validity in O(1) time
        #    by tracking how many distinct characters are "satisfied."
        
        from collections import defaultdict
        
        # Step 1: Compute frequency of each character in word2
        freq2 = [0] * 26
        for ch in word2:
            freq2[ord(ch) - ord('a')] += 1
        
        # Number of distinct characters in word2 that have a positive requirement
        neededDistinctCount = sum(1 for v in freq2 if v > 0)
        
        # If word2 has no characters (edge case) - not needed here as constraints say len(word2) >= 1
        # but we keep a guard
        if neededDistinctCount == 0:
            # Every substring of word1 is valid
            n = len(word1)
            return n * (n + 1) // 2
        
        # Step 2: Prepare sliding window trackers
        freq_window = [0] * 26
        currentSatisfied = 0  # How many distinct chars have freq_window[c] >= freq2[c]
        
        def add_char(ch):
            nonlocal currentSatisfied
            idx = ord(ch) - ord('a')
            old_satisfied = (freq_window[idx] >= freq2[idx]) if freq2[idx] > 0 else True
            freq_window[idx] += 1
            new_satisfied = (freq_window[idx] >= freq2[idx]) if freq2[idx] > 0 else True
            if freq2[idx] > 0 and old_satisfied != new_satisfied:
                if new_satisfied:
                    currentSatisfied += 1
                else:
                    currentSatisfied -= 1
        
        def remove_char(ch):
            nonlocal currentSatisfied
            idx = ord(ch) - ord('a')
            old_satisfied = (freq_window[idx] >= freq2[idx]) if freq2[idx] > 0 else True
            freq_window[idx] -= 1
            new_satisfied = (freq_window[idx] >= freq2[idx]) if freq2[idx] > 0 else True
            if freq2[idx] > 0 and old_satisfied != new_satisfied:
                if new_satisfied:
                    currentSatisfied += 1
                else:
                    currentSatisfied -= 1
        
        # Perform the sliding window
        n = len(word1)
        end = 0
        result = 0
        
        for start in range(n):
            # Expand end until the substring [start..end] is valid or we run out of string
            while end < n and currentSatisfied < neededDistinctCount:
                add_char(word1[end])
                end += 1
            
            # If we are valid at this point, all substrings starting at 'start'
            # and ending at [end..n-1] are valid
            if currentSatisfied == neededDistinctCount:
                result += (n - end + 1)
            
            # Remove the start character from the window as we move to the next start
            remove_char(word1[start])
        
        return result