class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        s_list = list(s)
        low, high = 0, len(s) - 1
        while low < high:
            if s_list[low] != s_list[high]:
                if s_list[low] < s_list[high]:
                    s_list[high] = s_list[low]
                else:
                    s_list[low] = s_list[high]
            low += 1
            high -= 1
        return ''.join(s_list)