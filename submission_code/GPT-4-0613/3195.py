class Solution:
    def minimumSteps(self, s: str) -> int:
        total_ones = s.count('1')
        current_ones = 0
        min_steps = float('inf')
        
        for i in range(len(s)):
            if s[i] == '1':
                current_ones += 1
            else:
                min_steps = min(min_steps, total_ones - current_ones)
                
        return min_steps if min_steps != float('inf') else 0