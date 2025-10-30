class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        res = ""
        for i in range(n - k + 1):
            if s[i] == '1' and (res == "" or ord(res[-1]) + 1 == ord(s[i])):
                if len(res) < k:
                    res += s[i]
                else:
                    if res < s[i:i+k]:
                        res = s[i:i+k]
        return res