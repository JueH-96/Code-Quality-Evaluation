class Solution:
    def isBalanced(self, num: str) -> bool:
        even_sum = 0
        odd_sum = 0
        for i, ch in enumerate(num):
            if i % 2 == 0:
                even_sum += int(ch)
            else:
                odd_sum += int(ch)
        return even_sum == odd_sum