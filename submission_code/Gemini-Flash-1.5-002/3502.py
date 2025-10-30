class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        count = 0
        n = len(s)
        if n == 0 or k > n:
            return 0

        for i in range(n):
            for j in range(i, n):
                substring = s[i:j+1]
                valid = False
                for char in set(substring):
                    if substring.count(char) >= k:
                        valid = True
                        break
                if valid:
                    count += 1
        return count