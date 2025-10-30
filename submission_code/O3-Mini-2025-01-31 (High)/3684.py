class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split the pattern into two parts using the '*' as the separator.
        prefix, suffix = p.split('*', 1)
        n = len(s)
        # Iterate over all possible non-empty contiguous substrings of s.
        for i in range(n):
            for j in range(i, n):
                candidate = s[i:j+1]
                # The candidate substring must be at least as long as the sum of the prefix and suffix.
                if len(candidate) < len(prefix) + len(suffix):
                    continue
                # Check if the candidate starts with the prefix and ends with the suffix.
                if candidate.startswith(prefix) and candidate.endswith(suffix):
                    return True
        return False

# Below is an example of how you might test the function:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    s1 = "leetcode"
    p1 = "ee*e"
    print(sol.hasMatch(s1, p1))  # Expected output: True
    
    # Example 2:
    s2 = "car"
    p2 = "c*v"
    print(sol.hasMatch(s2, p2))  # Expected output: False
    
    # Example 3:
    s3 = "luck"
    p3 = "u*"
    print(sol.hasMatch(s3, p3))  # Expected output: True