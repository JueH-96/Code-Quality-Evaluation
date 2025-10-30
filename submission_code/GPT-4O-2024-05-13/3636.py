class Solution:
    def isBalanced(self, num: str) -> bool:
        sum_even_indices = 0
        sum_odd_indices = 0
        
        for i in range(len(num)):
            if i % 2 == 0:
                sum_even_indices += int(num[i])
            else:
                sum_odd_indices += int(num[i])
        
        return sum_even_indices == sum_odd_indices