class Solution:
    def isBalanced(self, num: str) -> bool:
        even_sum = sum(int(digit) for digit in num[0::2])
        odd_sum = sum(int(digit) for digit in num[1::2])
        return even_sum == odd_sum