class Solution:
    def minimumSteps(self, s: str) -> int:
        black_count = 0
        steps = 0
        
        for char in s:
            if char == '1':
                black_count += 1
            else:
                steps += black_count
        
        return steps