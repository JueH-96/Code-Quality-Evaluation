class Solution:
    def minimumOperations(self, num: str) -> int:
        def find_min_ops(s, target):
            i, j = 0, 0
            while i < len(s) and j < len(target):
                if s[i] == target[j]:
                    j += 1
                i += 1
            return len(s) - (j == len(target) and j or 0)
        
        targets = ["00", "25", "50", "75"]
        min_ops = float('inf')
        
        for target in targets:
            min_ops = min(min_ops, find_min_ops(num, target))
        
        if '0' in num:
            min_ops = min(min_ops, len(num) - 1)
        
        return min_ops