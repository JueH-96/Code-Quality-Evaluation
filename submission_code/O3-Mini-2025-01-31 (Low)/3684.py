class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Split the pattern into two parts around the single '*'
        star_index = p.find('*')
        left = p[:star_index]
        right = p[star_index+1:]
        
        n = len(s)
        
        # Check every contiguous substring of s
        for i in range(n):
            for j in range(i, n):
                sub = s[i:j+1]
                # Check if length fits; if not, continue.
                if len(sub) < len(left) + len(right):
                    continue
                # Check if the substring starts with left and ends with right.
                if sub.startswith(left) and sub.endswith(right):
                    return True
        return False

# Sample Test Cases
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    s1, p1 = "leetcode", "ee*e"
    print(sol.hasMatch(s1, p1))  # Expected output: True

    # Example 2
    s2, p2 = "car", "c*v"
    print(sol.hasMatch(s2, p2))  # Expected output: False

    # Example 3
    s3, p3 = "luck", "u*"
    print(sol.hasMatch(s3, p3))  # Expected output: True

    # Additional Test: pattern is just '*'
    s4, p4 = "abc", "*"
    print(sol.hasMatch(s4, p4))  # Expected output: True (any non-empty substring matches)