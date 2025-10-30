class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Pointer for str2, to try to match its characters as a subsequence of str1.
        j = 0
        n, m = len(str1), len(str2)
        
        # Helper function to get the next cyclic character.
        def next_char(c: str) -> str:
            return 'a' if c == 'z' else chr(ord(c) + 1)
        
        # Traverse each character in str1.
        for i in range(n):
            if j >= m:
                break
            # For each character, we have two options:
            # 1. Leave it as is.
            # 2. Increment it (cyclic operation, 'z' becomes 'a').
            if str1[i] == str2[j] or next_char(str1[i]) == str2[j]:
                j += 1
                
        # If we matched all characters in str2 in order, return True.
        return j == m

# For example usage:
if __name__ == "__main__":
    solution = Solution()
    # Example test cases:
    print(solution.canMakeSubsequence("abc", "ad"))  # Expected output: True
    print(solution.canMakeSubsequence("zc", "ad"))   # Expected output: True
    print(solution.canMakeSubsequence("ab", "d"))    # Expected output: False