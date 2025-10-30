class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        # Calculate the positions where s1 and s2 differ
        diff_positions = []
        for i in range(n):
            if s1[i] != s2[i]:
                diff_positions.append(i)
        
        # If the number of differing positions is odd, it's impossible to make them equal
        if len(diff_positions) % 2 != 0:
            return -1
        
        # Initialize the cost
        cost = 0
        i = 0
        while i < len(diff_positions):
            # If two consecutive differing positions are adjacent, use the cheaper operation
            if i + 1 < len(diff_positions) and diff_positions[i + 1] == diff_positions[i] + 1:
                cost += 1
                i += 2  # Skip the next position as it's already handled
            else:
                # Otherwise, use the more expensive operation
                cost += x
                i += 2  # Handle two positions at a time
        
        return cost