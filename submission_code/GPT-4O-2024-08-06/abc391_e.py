def min_changes_to_flip(N, A):
    def majority(a, b, c):
        return (a + b + c) >= 2
    
    def recursive_majority(level, start, end):
        if level == 0:
            return int(A[start])
        
        step = 3 ** (level - 1)
        result = []
        for i in range(start, end, 3 * step):
            maj = majority(
                recursive_majority(level - 1, i, i + step),
                recursive_majority(level - 1, i + step, i + 2 * step),
                recursive_majority(level - 1, i + 2 * step, i + 3 * step)
            )
            result.append(maj)
        
        return result[0]
    
    def min_changes(level, start, end, target):
        if level == 0:
            return 0 if int(A[start]) == target else 1
        
        step = 3 ** (level - 1)
        changes = float('inf')
        
        for i in range(start, end, 3 * step):
            maj = majority(
                recursive_majority(level - 1, i, i + step),
                recursive_majority(level - 1, i + step, i + 2 * step),
                recursive_majority(level - 1, i + 2 * step, i + 3 * step)
            )
            
            if maj == target:
                continue
            
            change_count = (
                min_changes(level - 1, i, i + step, 1 - maj) +
                min_changes(level - 1, i + step, i + 2 * step, 1 - maj) +
                min_changes(level - 1, i + 2 * step, i + 3 * step, 1 - maj)
            )
            
            changes = min(changes, change_count)
        
        return changes
    
    current_value = recursive_majority(N, 0, len(A))
    target_value = 1 - current_value
    return min_changes(N, 0, len(A), target_value)

import sys
input = sys.stdin.read
data = input().strip().split()
N = int(data[0])
A = data[1]

print(min_changes_to_flip(N, A))