class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        n = len(s)
        count = 0

        # Iterate over each character in the string
        for i in range(n):
            if s[i] == c:
                # Count all substrings starting from s[i] and ending at any subsequent 'c'
                for j in range(i, n):
                    if s[j] == c:
                        count += 1

        return count