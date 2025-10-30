class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        # Initialize the cost and the current character
        cost = 0
        curr_char = s[0]
        
        # Iterate over the string
        for i in range(1, n):
            # If the current character is different from the previous one
            if s[i] != curr_char:
                # Update the cost and the current character
                cost += 1
                curr_char = s[i]
        
        # The minimum cost is the minimum between the cost of making all characters equal to '0' and the cost of making all characters equal to '1'
        return min(cost, n - cost)