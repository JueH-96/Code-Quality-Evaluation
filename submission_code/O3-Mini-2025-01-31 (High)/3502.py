class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # frequency array for 26 lowercase letters
        freq = [0] * 26
        # Keeps track of how many characters in the current window have frequency >= k.
        count_at_least_k = 0
        ans = 0
        r = 0
        
        # Iterate with left pointer of the window.
        for l in range(n):
            # Expand the window until it becomes valid (i.e. at least one character appears k times)
            while r < n and count_at_least_k == 0:
                idx = ord(s[r]) - ord('a')
                freq[idx] += 1
                # Update count_at_least_k when a character reaches exactly k occurrences.
                if freq[idx] == k:
                    count_at_least_k += 1
                r += 1

            # If the window is valid then every substring starting at l and ending from r to n-1 is valid.
            if count_at_least_k > 0:
                ans += (n - r + 1)
            
            # Before moving the left pointer, remove the leftmost character from the window.
            idx = ord(s[l]) - ord('a')
            if freq[idx] == k:
                count_at_least_k -= 1
            freq[idx] -= 1

        return ans