class Solution:
    def minimumSteps(self, s: str) -> int:
        steps = 0
        white_count = 0
        
        # Traverse from right to left
        for i in range(len(s) - 1, -1, -1):
            if s[i] == '0':
                # Count white balls as we go from right to left
                white_count += 1
            else:
                # For each black ball, add the number of white balls to its right
                steps += white_count
        
        return steps