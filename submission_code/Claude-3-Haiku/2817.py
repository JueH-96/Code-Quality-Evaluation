class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        cost = 0
        
        # Count the number of 0s and 1s
        count_0 = s.count('0')
        count_1 = n - count_0
        
        # If all characters are the same, return 0
        if count_0 == 0 or count_1 == 0:
            return 0
        
        # Compute the cost for inverting from the left
        cost_left = 0
        for i in range(n):
            if s[i] == '0':
                cost_left += i + 1
            else:
                cost_left += n - i
        
        # Compute the cost for inverting from the right
        cost_right = 0
        for i in range(n - 1, -1, -1):
            if s[i] == '1':
                cost_right += i + 1
            else:
                cost_right += n - i
        
        # Return the minimum cost
        return min(cost_left, cost_right)