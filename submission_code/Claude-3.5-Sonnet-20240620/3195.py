class Solution:
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        steps = 0
        black_count = 0
        
        for i in range(n):
            if s[i] == '1':
                black_count += 1
            else:
                steps += black_count
        
        return steps