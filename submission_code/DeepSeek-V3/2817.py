class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        if n == 1:
            return 0
        
        # Calculate the cost to make all characters '0'
        cost0 = 0
        current = '0'
        for i in range(n):
            if s[i] != current:
                cost0 += i + 1
                current = '1' if current == '0' else '0'
        
        # Calculate the cost to make all characters '1'
        cost1 = 0
        current = '1'
        for i in range(n):
            if s[i] != current:
                cost1 += i + 1
                current = '1' if current == '0' else '0'
        
        # Also, consider the cost of flipping from the end
        # Reset and calculate cost0 from the end
        cost0_end = 0
        current = '0'
        for i in range(n-1, -1, -1):
            if s[i] != current:
                cost0_end += n - i
                current = '1' if current == '0' else '0'
        
        # Reset and calculate cost1 from the end
        cost1_end = 0
        current = '1'
        for i in range(n-1, -1, -1):
            if s[i] != current:
                cost1_end += n - i
                current = '1' if current == '0' else '0'
        
        # The minimum cost is the minimum of all four options
        return min(cost0, cost1, cost0_end, cost1_end)