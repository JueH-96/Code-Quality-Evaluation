class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        # If the string is shorter than 2, no substrings of length 2 exist
        if len(s) < 2:
            return False
        
        rev_s = s[::-1]
        # Check every substring of length 2 in s
        for i in range(len(s) - 1):
            sub = s[i:i+2]
            if sub in rev_s:
                return True
        return False

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.isSubstringPresent("leetcode"))  # True
    print(sol.isSubstringPresent("abcba"))     # True
    print(sol.isSubstringPresent("abcd"))      # False