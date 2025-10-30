class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        
        # If the string length is 0, return 0 (though constraints state n >= 1).
        # If n is 1, the loop below correctly handles it, resulting in max_len = 1.
        # For small n (e.g., n=2), the loop also works correctly (e.g., "11" -> max_len=2).
        # So, no special base cases for small n are strictly necessary.
        if n == 0:
            return 0

        left = 0
        pair_count = 0  # Counts consecutive identical pairs (e.g., "AA") within the current window s[left:right+1]
        max_len = 0     # Stores the maximum length of a semi-repetitive substring found so far

        # Iterate 'right' pointer to expand the window
        for right in range(n):
            # When expanding the window, check if the newly included character s[right]
            # forms a consecutive pair with the character just before it, s[right-1].
            if right > 0 and s[right] == s[right-1]:
                pair_count += 1
            
            # If the current window (s[left:right+1]) has more than one
            # consecutive pair, it's not semi-repetitive.
            # We need to shrink the window from the left by advancing 'left'
            # until 'pair_count' is 1 or 0 (i.e., the window becomes semi-repetitive again).
            while pair_count > 1:
                # As 'left' advances, if the character at 'left' formed a pair
                # with the character at 'left+1', that specific pair is now
                # outside the window. So, we decrement pair_count.
                # It's safe to access s[left+1] because if pair_count > 1, it implies
                # there are at least two pairs in the window, meaning the window length
                # is at least 4, ensuring left+1 is a valid index within the original string.
                if s[left] == s[left+1]:
                    pair_count -= 1
                left += 1
            
            # At this point, the current window s[left:right+1] is guaranteed to be semi-repetitive.
            # Calculate its length and update max_len if this window is longer.
            max_len = max(max_len, right - left + 1)
        
        return max_len