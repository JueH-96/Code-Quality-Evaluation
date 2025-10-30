class Solution:
    def isBalanced(self, num: str) -> bool:
        even_sum = 0
        odd_sum = 0
        for i, c in enumerate(num):
            if i % 2 == 0:
                even_sum += int(c)
            else:
                odd_sum += int(c)
        return even_sum == odd_sum