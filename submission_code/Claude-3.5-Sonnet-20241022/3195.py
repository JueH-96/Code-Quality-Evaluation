class Solution:
    def minimumSteps(self, s: str) -> int:
        steps = 0
        count_ones = 0
        
        # Count steps needed to move each 1 to the right
        for i in range(len(s)):
            if s[i] == '1':
                count_ones += 1
            else:
                # For each 0, we need to move all previous 1s one step right
                steps += count_ones
                
        return steps