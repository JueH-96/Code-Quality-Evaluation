class Solution:
    def isBalanced(self, num: str) -> bool:
        sum_even = 0
        sum_odd = 0
        for i, c in enumerate(num):
            digit = int(c)
            if i % 2 == 0:
                sum_even += digit
            else:
                sum_odd += digit
        return sum_even == sum_odd