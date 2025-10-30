class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        """
        We say a substring of word1 is "valid" if we can rearrange it so that word2
        appears as a prefix. Equivalently, the substring must contain at least as
        many of each character as word2 does (i.e. it is a 'super-multiset' of word2).

        To count these efficiently, we use a two-pointer (sliding-window) approach
        combined with frequency counts:

        1. Let freq2[] be the frequency of each character in word2.
           Let needed = sum(freq2), the total number of characters in word2.

        2. We scan word1 from left to right with an expanding 'right' pointer.
           Maintain freq1[] for the window [left..right] and a variable 'have'
           that keeps track of how many total characters (across all letters)
           from word2 we have "covered" so far in freq1[].
           Concretely, have = sum(min(freq1[c], freq2[c]) for each character c).

        3. Whenever we add a character word1[right], we update freq1[] and
           adjust 'have' accordingly:
               old = min(freq1[x], freq2[x])  # before increment
               freq1[x] += 1
               new = min(freq1[x], freq2[x])  # after increment
               have += (new - old)

        4. Then we try to shrink from the left if we have any "excess" characters 
           that exceed what is needed by word2.  In other words, while
               freq1[word1[left]] > freq2[word1[left]],
           we decrement freq1[word1[left]] and move left forward.  Note that
           in this "excess-shrinking" step, 'have' does not change, because removing 
           an excess count does not affect the portion that actually covers word2.

        5. If after this shrinking step 'have' == needed, then the current window 
           [left..right] contains at least all the characters in word2.  Any larger
           start index < left would make an even bigger window, which still contains 
           all of word2.  So the minimal valid start is 'left'.  Therefore we add
           (left + 1) to our running result (the number of ways to choose the start
           index among 0..left).

        6. We continue for each right until the end of word1.

        This algorithm runs in O(n) time, where n = len(word1), because
        each pointer (left and right) moves at most n steps total.
        """

        # Quick edge case: if word2 is longer than word1, no substring of word1 can contain all of word2.
        if len(word2) > len(word1):
            return 0

        # Frequency array for 'a'..'z'.
        freq2 = [0]*26
        for ch in word2:
            freq2[ord(ch) - ord('a')] += 1
        
        # Total number of characters we must cover.
        needed = len(word2)
        
        # Sliding window frequency and 'have' counter.
        freq1 = [0]*26
        have = 0
        
        result = 0
        left = 0
        
        # Helper to update 'have' when we increment freq1[x].
        def add_char(x):
            nonlocal have
            old = min(freq1[x], freq2[x])
            freq1[x] += 1
            new = min(freq1[x], freq2[x])
            have += (new - old)
        
        # Process each character with the 'right' pointer.
        for right, ch in enumerate(word1):
            idx = ord(ch) - ord('a')
            add_char(idx)
            
            # Shrink from the left if we have excess of the leftmost character.
            while freq1[ord(word1[left]) - ord('a')] > freq2[ord(word1[left]) - ord('a')]:
                freq1[ord(word1[left]) - ord('a')] -= 1
                left += 1
            
            # If we fully cover all characters from word2, count possible starts.
            if have == needed:
                # 'left' is the minimal start index that still contains word2,
                # so all starts in [0..left] also contain word2.
                result += (left + 1)
        
        return result