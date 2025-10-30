class Solution:
    def isBalanced(self, num: str) -> bool:
        even_sum = 0
        odd_sum = 0
        for i, c in enumerate(num):
            digit = int(c)
            if i % 2 == 0:
                even_sum += digit
            else:
                odd_sum += digit
        return even_sum == odd_sum