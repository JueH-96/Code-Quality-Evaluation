from collections import deque

class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x == y:
            return 0
        
        lower_bound = 1
        upper_bound = 20000
        
        queue = deque()
        queue.append(x)
        visited = set()
        visited.add(x)
        steps = {x: 0}
        
        while queue:
            num = queue.popleft()
            current_steps = steps[num]
            
            if num == y:
                return current_steps
            
            next_states = []
            if num % 11 == 0:
                next_num = num // 11
                if lower_bound <= next_num <= upper_bound:
                    next_states.append(next_num)
            if num % 5 == 0:
                next_num = num // 5
                if lower_bound <= next_num <= upper_bound:
                    next_states.append(next_num)
            next_num = num - 1
            if lower_bound <= next_num <= upper_bound:
                next_states.append(next_num)
            next_num = num + 1
            if lower_bound <= next_num <= upper_bound:
                next_states.append(next_num)
                
            for state in next_states:
                if state not in visited:
                    visited.add(state)
                    steps[state] = current_steps + 1
                    queue.append(state)
                    
        return -1