class Solution:
    def minimumOperations(self, num: str) -> int:
        targets = ['00', '25', '50', '75']
        min_ops = float('inf')
        
        for target in targets:
            target_0 = target[0]
            target_1 = target[1]
            
            # Find the rightmost occurrence of target_1
            j = -1
            for idx in range(len(num)-1, -1, -1):
                if num[idx] == target_1:
                    j = idx
                    break
            if j == -1:
                continue
            
            # Find the rightmost occurrence of target_0 before j
            i = -1
            for idx in range(j-1, -1, -1):
                if num[idx] == target_0:
                    i = idx
                    break
            if i == -1:
                continue
            
            # Calculate deletions
            deletions = len(num) - (i + 2)
            if deletions < min_ops:
                min_ops = deletions
        
        # Check for the case where the string contains at least one '0'
        if '0' in num:
            possible_ops = len(num) - 1
            if possible_ops < min_ops:
                min_ops = possible_ops
        
        # Handle the case where no targets are found and no '0' in the string
        if min_ops == float('inf'):
            if '0' in num:
                min_ops = len(num) - 1
            else:
                min_ops = len(num)
        
        return min_ops