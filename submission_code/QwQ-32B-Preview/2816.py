class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s_list = list(s)
        n = len(s)
        for i in range(n // 2):
            left = s_list[i]
            right = s_list[n - 1 - i]
            if left != right:
                smaller = min(left, right)
                s_list[i] = smaller
                s_list[n - 1 - i] = smaller
        return ''.join(s_list)