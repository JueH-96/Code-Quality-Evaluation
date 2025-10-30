class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                sub = s[i : j + 1]
                zeros = sub.count("0")
                ones = sub.count("1")
                if ones >= zeros * zeros:
                    ans += 1
        return ans