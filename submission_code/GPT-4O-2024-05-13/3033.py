class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        
        # Find the positions where s1 and s2 differ
        diff_positions = [i for i in range(n) if s1[i] != s2[i]]
        
        # If the number of differing positions is odd, it's impossible to make them equal
        if len(diff_positions) % 2 != 0:
            return -1
        
        # Initialize the cost
        cost = 0
        
        # Try to use the cheaper operation (cost 1) as much as possible
        i = 0
        while i < len(diff_positions) - 1:
            if diff_positions[i + 1] == diff_positions[i] + 1:
                # If two differing positions are consecutive, use the cheaper operation
                cost += 1
                i += 2
            else:
                i += 1
        
        # Calculate the remaining pairs to be fixed with the more expensive operation
        remaining_pairs = (len(diff_positions) - i) // 2
        cost += remaining_pairs * x
        
        return cost