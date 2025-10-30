from typing import List

class Solution:
    def maxWeight(self, pizzas: List[int]) -> int:
        # sort all pizzas so that we can always pick
        #   – the currently biggest ones from the right end
        #   – the currently smallest ones from the left end
        pizzas.sort()
        
        n = len(pizzas)
        left, right = 0, n - 1          # pointers to the current smallest / biggest
        day = 1                         # 1-indexed day counter
        gain = 0
        
        # we remove 4 pizzas per day, therefore the loop runs n//4 times
        while left <= right:
            if day & 1:                 # odd-numbered day – we keep Z (largest)
                gain += pizzas[right]   # Z
                right -= 1              # one largest pizza is eaten
                left  += 3              # together with three smallest ones
            else:                       # even-numbered day – we keep Y (2nd largest)
                gain += pizzas[right - 1]   # Y (second largest of the 4)
                right -= 2                  # two largest pizzas are eaten
                left  += 2                  # together with two smallest ones
            day += 1
        
        return gain