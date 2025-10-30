class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        res = 0
        # frequency dictionary for current window and count of characters meeting the k threshold.
        freq = {}
        count_valid = 0  # number of characters in current window with frequency >= k
        
        # Helper function to update frequency when adding a character.
        def add(ch):
            nonlocal count_valid
            freq[ch] = freq.get(ch, 0) + 1
            # When frequency reaches exactly k, we add to count_valid.
            if freq[ch] == k:
                count_valid += 1
        
        # Helper function to update frequency when removing a character.
        def remove(ch):
            nonlocal count_valid
            # Before removal, if frequency is k, removal will break the condition.
            if freq[ch] == k:
                count_valid -= 1
            freq[ch] -= 1
            if freq[ch] == 0:
                del freq[ch]
        
        # Two pointers: window is s[left:right] (right is exclusive).
        right = 0
        for left in range(n):
            # Expand right until the window has at least one character with frequency >= k.
            while right < n and count_valid == 0:
                add(s[right])
                right += 1
                
            # If we've achieved a valid window, then any extension of the window will remain valid.
            if count_valid > 0:
                res += (n - right + 1)
            else:
                # If not valid and right reached the end, then no further windows starting at left can be valid.
                break
            
            # Remove the leftmost character before moving left pointer.
            remove(s[left])
        
        return res