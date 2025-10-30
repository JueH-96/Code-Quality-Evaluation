class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s_list = list(s)
        n = len(s_list)
        for i in range(n // 2):
            j = n - 1 - i
            if s_list[i] != s_list[j]:
                if s_list[i] < s_list[j]:
                    # Change s[j] to s[i]
                    s_list[j] = s_list[i]
                else:
                    # Change s[i] to s[j]
                    s_list[i] = s_list[j]
        return ''.join(s_list)