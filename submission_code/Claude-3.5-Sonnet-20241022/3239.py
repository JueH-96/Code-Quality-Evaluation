class Solution:
    def minimumOperationsToMakeEqual(self, x: int, y: int) -> int:
        if x <= y:
            return y - x
            
        seen = set()
        queue = [(x, 0)]
        seen.add(x)
        
        while queue:
            curr, steps = queue.pop(0)
            
            if curr == y:
                return steps
                
            # Try all possible operations
            operations = []
            
            # Divide by 11 if multiple of 11
            if curr % 11 == 0:
                operations.append(curr // 11)
                
            # Divide by 5 if multiple of 5    
            if curr % 5 == 0:
                operations.append(curr // 5)
                
            # Increment and decrement
            operations.append(curr + 1)
            operations.append(curr - 1)
            
            for next_num in operations:
                if next_num not in seen and next_num >= 1:
                    seen.add(next_num)
                    queue.append((next_num, steps + 1))
        
        return -1