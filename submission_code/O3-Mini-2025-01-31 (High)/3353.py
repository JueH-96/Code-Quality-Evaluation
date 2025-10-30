class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # If length of s is less than 2, then no substring of length 2 exists.
        if len(s) < 2:
            return False
        
        # Compute the reverse of the string.
        rev = s[::-1]
        
        # Iterate over all possible substrings of length 2 in s.
        for i in range(len(s) - 1):
            # Extract substring of length 2.
            substr = s[i:i+2]
            # Check if the substring exists in the reversed string.
            if substr in rev:
                return True
        
        # If no substring is found, return false.
        return False

# Example usage:
solution = Solution()
print(solution.isSubstringPresent("leetcode"))  # Expected output: True
print(solution.isSubstringPresent("abcba"))     # Expected output: True
print(solution.isSubstringPresent("abcd"))      # Expected output: False