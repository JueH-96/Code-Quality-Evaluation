class Solution:
    def isBalanced(self, num: str) -> bool:
        even_sum = 0
        odd_sum = 0
        
        for i in range(len(num)):
            digit = int(num[i])
            if i % 2 == 0:  # Even index (0, 2, 4, ...)
                even_sum += digit
            else:  # Odd index (1, 3, 5, ...)
                odd_sum += digit
        
        return even_sum == odd_sum