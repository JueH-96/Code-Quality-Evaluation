class Solution:
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        count_black = 0
        steps = 0
        
        # Count the number of black balls
        for c in s:
            if c == '1':
                count_black += 1
        
        # If all balls are black or all balls are white, no steps are needed
        if count_black == 0 or count_black == n:
            return 0
        
        # Iterate through the string and swap adjacent balls to group the black balls to the right
        i = 0
        while i < n - 1:
            if s[i] == '1' and s[i+1] == '0':
                s = s[:i] + '01' + s[i+2:]
                steps += 1
                i += 2
            else:
                i += 1
        
        return steps