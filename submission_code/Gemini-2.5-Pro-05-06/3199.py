class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        
        # Iterate over possible numbers of candies for the first child (c1)
        # c1 must be between 0 and limit (inclusive).
        # Also, c1 cannot be greater than n (total candies), as c2 and c3 must be non-negative.
        # So, c1 ranges from 0 to min(n, limit).
        # The '+1' makes the range inclusive of the upper bound min(n, limit).
        for c1 in range(min(n, limit) + 1):
            # Iterate over possible numbers of candies for the second child (c2)
            # c2 must be between 0 and limit (inclusive).
            # Also, c2 cannot be greater than n - c1 (remaining candies after c1's share),
            # as c3 must be non-negative.
            # So, c2 ranges from 0 to min(n - c1, limit).
            # Note: n - c1 is guaranteed to be non-negative because c1 <= n from the first loop's condition.
            for c2 in range(min(n - c1, limit) + 1):
                # The number of candies for the third child (c3) is now determined.
                c3 = n - c1 - c2
                
                # We need to check if this distribution (c1, c2, c3) is valid.
                # The loop bounds for c1 and c2 ensure:
                #   0 <= c1 <= limit
                #   0 <= c2 <= limit
                # The calculation of c3 ensures:
                #   c1 + c2 + c3 = n
                # We also need to ensure c3 >= 0.
                #   Since c2 <= n - c1 (from the second loop's condition),
                #   we have c1 + c2 <= n.
                #   Therefore, c3 = n - (c1 + c2) >= 0. This is guaranteed.
                
                # The only remaining condition to check is whether c3 exceeds the limit.
                if c3 <= limit:
                    count += 1
                    
        return count