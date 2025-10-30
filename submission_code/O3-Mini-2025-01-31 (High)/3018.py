class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Two-pointer approach:
        # For each character in str1, we have two options:
        # 1. Keep the character as is.
        # 2. Increment the character (cyclically: 'z' becomes 'a').
        # We try to match str2 sequentially by checking if either option
        # equals the current target character.
        j = 0  # pointer for str2
        m = len(str2)
        
        # Helper function to get the next character cyclically.
        def next_char(c: str) -> str:
            return 'a' if c == 'z' else chr(ord(c) + 1)
        
        for c in str1:
            if j < m and (c == str2[j] or next_char(c) == str2[j]):
                j += 1
                if j == m:
                    break
        return j == m

# Optional testing code:
if __name__ == "__main__":
    sol = Solution()
    # Example 1: "abc" can be turned into "abd" by incrementing the last letter.
    print(sol.canMakeSubsequence("abc", "ad"))  # Expected output: True
    
    # Example 2: "zc" can be turned into "ad" by incrementing both letters.
    print(sol.canMakeSubsequence("zc", "ad"))   # Expected output: True
    
    # Example 3: "ab" cannot yield subsequence "d" regardless of increments.
    print(sol.canMakeSubsequence("ab", "d"))    # Expected output: False