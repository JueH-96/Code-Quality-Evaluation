class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        """
        Calculates the total number of ways to distribute n candies among 3 children
        such that no child gets more than limit candies.
        """
        
        # We need to find the number of non-negative integer solutions for
        # c1 + c2 + c3 = n, where c1, c2, c3 <= limit.
        
        # We can iterate through all valid possibilities for the first two children
        # (c1 and c2) and then determine if the third child's share (c3) is valid.
        
        count = 0
        
        # Iterate through possible candies for the first child (c1).
        # c1 can range from 0 to 'limit'. Also, c1 cannot exceed 'n'.
        for c1 in range(min(n, limit) + 1):
            
            # Iterate through possible candies for the second child (c2).
            # c2 can range from 0 to 'limit'.
            # Also, c1 + c2 cannot exceed 'n', so c2 <= n - c1.
            for c2 in range(min(n - c1, limit) + 1):
                
                # The number of candies for the third child (c3) is now determined.
                c3 = n - c1 - c2
                
                # We need to check if c3 is within the valid range [0, limit].
                # From the loop bounds, c3 >= 0 is already guaranteed because
                # c2 <= n - c1 => c1 + c2 <= n => n - c1 - c2 >= 0.
                # We only need to check if c3 <= limit.
                if c3 <= limit:
                    count += 1
                    
        return count