class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n = len(word1)
        # Build frequency requirement for word2 (only lowercase letters assumed)
        req = [0] * 26
        for ch in word2:
            req[ord(ch) - 97] += 1
        # Only consider letters that are required (non-zero count in word2)
        needed = [i for i in range(26) if req[i] > 0]
        
        # This window array will hold frequency counts for the current substring in word1.
        window = [0] * 26
        
        # Helper function to check if the current window (substring) meets the required counts.
        def is_valid():
            for idx in needed:
                if window[idx] < req[idx]:
                    return False
            return True
        
        total = 0
        r = 0  # r will be the end pointer (one past the last character in the window)
        
        # For each starting index i, we try to find the minimal j (represented by r-1)
        # for which the substring word1[i:j+1] (i.e. word1[i:r]) can be rearranged to have word2 as prefix.
        # In other words, it needs to contain at least the frequency counts given by req.
        #
        # Note: If word1[i:r] is valid then any substring word1[i:j+1] with j >= r-1 is also valid 
        # because it contains all the required characters (and possibly more).
        for i in range(n):
            # Extend the window until it becomes valid or we reach the end.
            while r < n and not is_valid():
                window[ord(word1[r]) - 97] += 1
                r += 1
            
            # If current window (from i to r-1) is valid then all substrings starting at i that end
            # at any index from r-1 to n-1 are valid.
            if is_valid():
                total += (n - r + 1)
            else:
                # If even the window from i to the end of word1 isn’t valid, then no later windows will be.
                break
            
            # Before moving to the next starting index, remove word1[i] from the window.
            window[ord(word1[i]) - 97] -= 1
            
        return total

# For testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.validSubstringCount("bcca", "abc"))      # Output: 1

    # Example 2:
    print(sol.validSubstringCount("abcabc", "abc"))    # Output: 10

    # Example 3:
    print(sol.validSubstringCount("abcabc", "aaabc"))  # Output: 0