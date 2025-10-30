class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)
        # For a substring to be valid it must be at least as long as word2.
        if m > n:
            return 0
        
        # Build frequency array for word2, using 26 letters.
        target = [0] * 26
        for c in word2:
            target[ord(c) - 97] += 1
        
        # We'll use a sliding window technique.
        # window_freq holds counts for the current window.
        window_freq = [0] * 26
        
        # missing counts how many characters (total count) are still needed for the window
        # to fulfill the target requirement.
        missing = sum(target)
        
        total_valid = 0
        j = 0
        # Iterate over possible starting indices i.
        for i in range(n):
            # Expand the window until the substring [i, j-1] contains at least the required counts.
            while j < n and missing > 0:
                idx = ord(word1[j]) - 97
                # If this character is still needed then contribute to fulfilling the missing count.
                if window_freq[idx] < target[idx]:
                    missing -= 1
                window_freq[idx] += 1
                j += 1
            
            # If after expanding the window, missing becomes 0,
            # then the current window [i, j-1] is the smallest valid substring starting at index i.
            if missing == 0:
                # Every substring that starts at i and ends at k, where k ranges from j-1 to n-1 is valid.
                total_valid += (n - j + 1)
            
            # Now move the starting pointer i: remove word1[i] from the window.
            idx = ord(word1[i]) - 97
            # If the letter at i was critical in meeting the target (i.e. its frequency was <= target)
            # then by removing it, we now become deficient in that letter.
            window_freq[idx] -= 1
            if window_freq[idx] < target[idx]:
                missing += 1
        
        return total_valid

# Example test cases
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    word1 = "bcca"
    word2 = "abc"
    print(sol.validSubstringCount(word1, word2))  # Expected output: 1

    # Example 2:
    word1 = "abcabc"
    word2 = "abc"
    print(sol.validSubstringCount(word1, word2))  # Expected output: 10

    # Example 3:
    word1 = "abcabc"
    word2 = "aaabc"
    print(sol.validSubstringCount(word1, word2))  # Expected output: 0