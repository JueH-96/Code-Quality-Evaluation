class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        # If word2 is longer than word1, no substring of word1 can contain all chars of word2
        if len(word2) > len(word1):
            return 0
        
        # Frequency array of word2
        freq2 = [0]*26
        for ch in word2:
            freq2[ord(ch) - ord('a')] += 1
        
        # Quickly check if word2 has total chars more than length of word1 (impossible match)
        if sum(freq2) > len(word1):
            return 0
        
        # Sliding window approach
        freqWindow = [0]*26
        # Count how many distinct characters are "not yet satisfied" 
        # (i.e., window does not have enough of those chars to match freq2[c])
        unsatisfied = 0
        for c in range(26):
            if freq2[c] > 0:
                unsatisfied += 1
        
        # Helpers to update frequencies and adjust unsatisfied
        def inc(idx):
            nonlocal unsatisfied
            old_count = freqWindow[idx]
            freqWindow[idx] += 1
            # If freq2[idx] > 0 and we moved from (freq2[idx]-1) to freq2[idx], 
            # we've just satisfied this character
            if freq2[idx] > 0 and old_count == freq2[idx] - 1:
                unsatisfied -= 1
        
        def dec(idx):
            nonlocal unsatisfied
            old_count = freqWindow[idx]
            freqWindow[idx] -= 1
            # If freq2[idx] > 0 and we moved from freq2[idx] to (freq2[idx]-1),
            # we've just broken satisfaction for this character
            if freq2[idx] > 0 and old_count == freq2[idx]:
                unsatisfied += 1

        n = len(word1)
        ans = 0
        left = 0
        
        # Expand right pointer
        for right in range(n):
            inc(ord(word1[right]) - ord('a'))
            
            # While we have a valid window (all needed chars satisfied),
            # every extension of this window to the right is valid
            while unsatisfied == 0:
                ans += (n - right)
                dec(ord(word1[left]) - ord('a'))
                left += 1
        
        return ans