from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def can_swap_to(x, y):
            if x == y:
                return True
            num_x = str(x)
            num_y = str(y)
            
            # Swap in x
            for i in range(len(num_x)):
                for j in range(i + 1, len(num_x)):
                    swapped_x = list(num_x)
                    swapped_x[i], swapped_x[j] = swapped_x[j], swapped_x[i]
                    swapped_x_int = int(''.join(swapped_x))
                    if swapped_x_int == y:
                        return True
            
            # Swap in y
            for i in range(len(num_y)):
                for j in range(i + 1, len(num_y)):
                    swapped_y = list(num_y)
                    swapped_y[i], swapped_y[j] = swapped_y[j], swapped_y[i]
                    swapped_y_int = int(''.join(swapped_y))
                    if swapped_y_int == x:
                        return True
            
            return False
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if can_swap_to(nums[i], nums[j]):
                    count += 1
        return count