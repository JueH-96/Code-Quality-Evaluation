class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s_list = list(s)
        n = len(s_list)
        for i in range(n // 2):
            j = n - 1 - i
            if s_list[i] != s_list[j]:
                target = min(s_list[i], s_list[j])
                s_list[i] = target
                s_list[j] = target
        return ''.join(s_list)