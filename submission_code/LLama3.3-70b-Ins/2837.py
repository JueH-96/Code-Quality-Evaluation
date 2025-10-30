from collections import deque

class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        if num2 >= 0:
            return -1 if num1 < num2 else (num1 - num2 + num2 - 1) // num2 + 1 if num2 != 0 else -1
        
        queue = deque([(num1, 0)])
        visited = set([num1])
        
        while queue:
            curr_num, steps = queue.popleft()
            
            if curr_num == 0:
                return steps
            
            for i in range(61):
                next_num = curr_num - (2 ** i + num2)
                
                if next_num not in visited and next_num >= 0:
                    queue.append((next_num, steps + 1))
                    visited.add(next_num)
                    
        return -1