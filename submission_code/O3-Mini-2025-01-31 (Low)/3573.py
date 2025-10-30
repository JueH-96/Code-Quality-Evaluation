class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        
        n = len(word1)
        m = len(word2)
        # Frequency required from word2
        req = Counter(word2)
        
        # If word2 is longer than word1, no valid substring exists.
        if m > n:
            return 0
        
        total = 0
        have = Counter()
        # Two pointers, r is the right pointer of sliding window
        r = 0
        
        # Helper function to check if current window (have) meets requirements (req)
        def meets_requirements():
            # For each char required check available frequency in current window
            for ch, count in req.items():
                if have[ch] < count:
                    return False
            return True
        
        # We use a two pointer method; For each left index, find the minimal right index
        for l in range(n):
            # extend right pointer until we have a valid window or we reached the end
            while r < n and not meets_requirements():
                have[word1[r]] += 1
                r += 1
            # if after while, the window doesn't meet requirements, then no window starting from l can be valid
            if not meets_requirements():
                break  # because r is at end and window not valid, increasing l will only shrink window
            # When meets_requirements() is True, all substrings from current l ending at indices from r-1 to n-1 are valid.
            total += (n - r + 1)
            
            # Now move l forward by removing one character from the left of window
            have[word1[l]] -= 1
            
        return total


# For simple testing
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.validSubstringCount("bcca", "abc"))  # Expected output: 1
    # Example 2
    print(sol.validSubstringCount("abcabc", "abc"))  # Expected output: 10
    # Example 3
    print(sol.validSubstringCount("abcabc", "aaabc"))  # Expected output: 0