class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # Helper function to compute the total sum of the first k natural numbers
        def total_sum(k):
            return k * (k + 1) // 2
        
        # Helper function to compute the sum of the "odd-index" rows if row #1 is odd-index
        # For k rows:
        #  - If k is even -> odd indices = k/2 terms, sum of first n odd numbers = n^2
        #  - If k is odd -> odd indices = (k+1)/2 terms, sum of first n odd numbers = n^2
        def sum_odd_indices(k):
            if k % 2 == 0:
                return (k // 2) ** 2
            else:
                return ((k + 1) // 2) ** 2
        
        max_height = 0
        
        # Because red + blue <= 200 at most (from constraints), we won't exceed k ~ 19
        # (since 20*21/2 = 210 > 200). We'll just iterate up to a bit above that.
        for k in range(1, 201):
            # If total balls required for k rows is more than we can have, break early
            if total_sum(k) > red + blue:
                break
            
            # sum of odd index rows if top row is red
            odd_sum = sum_odd_indices(k)
            even_sum = total_sum(k) - odd_sum
            
            # Scenario A: top row red => red uses odd_sum, blue uses even_sum
            scenarioA = (odd_sum <= red and even_sum <= blue)
            
            # Scenario B: top row blue => blue uses odd_sum, red uses even_sum
            scenarioB = (odd_sum <= blue and even_sum <= red)
            
            if scenarioA or scenarioB:
                max_height = k
        
        return max_height