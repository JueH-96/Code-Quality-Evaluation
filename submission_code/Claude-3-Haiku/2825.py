class Solution:
    def minimizedStringLength(self, s: str) -> int:
        while True:
            min_length = len(s)
            for i in range(len(s)):
                left = s.rfind(s[i], 0, i)
                right = s.find(s[i], i+1)
                if left != -1 and right != -1:
                    s = s[:left] + s[left+1:right] + s[right+1:]
                    break
            else:
                return min_length