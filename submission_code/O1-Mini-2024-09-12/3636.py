class Solution:
    def isBalanced(self, num: str) -> bool:
        sum_even = 0
        sum_odd = 0
        for idx, digit in enumerate(num):
            if idx % 2 == 0:
                sum_even += int(digit)
            else:
                sum_odd += int(digit)
        return sum_even == sum_odd