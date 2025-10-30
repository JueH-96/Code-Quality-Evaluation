class Solution:
    def minimumOperations(self, num: str) -> int:
        min_ops = len(num)  # Initialize with the maximum possible steps (delete all)
        
        # List of possible two-digit endings
        targets = ['00', '25', '50', '75']
        
        for target in targets:
            x = target[0]
            y = target[1]
            
            # Find the rightmost occurrence of y
            y_pos = -1
            for i in range(len(num)-1, -1, -1):
                if num[i] == y:
                    y_pos = i
                    break
            if y_pos == -1:
                continue  # can't form this target
            
            # Find the rightmost occurrence of x before y_pos
            x_pos = -1
            for i in range(y_pos-1, -1, -1):
                if num[i] == x:
                    x_pos = i
                    break
            if x_pos == -1:
                continue  # can't form this target
            
            # Calculate the steps
            steps = (len(num) - y_pos - 1) + (y_pos - x_pos - 1)
            if steps < min_ops:
                min_ops = steps
        
        # Check if we can make the number a single '0'
        if '0' in num:
            steps_single_zero = len(num) - 1
            if steps_single_zero < min_ops:
                min_ops = steps_single_zero
        
        return min_ops