class Solution:
    def makeStringGood(self, s: str) -> int:
        counts = {}
        for char in s:
            counts[char] = counts.get(char, 0) + 1
        
        values = list(counts.values())
        n = len(values)
        
        min_ops = float('inf')
        
        for target_count in values:
            ops = 0
            for count in values:
                if count < target_count:
                    ops += count
                elif count > target_count:
                    ops += count - target_count
            min_ops = min(min_ops, ops)
            
        return min_ops