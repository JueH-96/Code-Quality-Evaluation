class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        
        # Iterate through possible number of candies for child 1 (c1)
        # c1 must be non-negative.
        # c1 cannot exceed 'limit'.
        # c1 cannot exceed 'n' (total candies), as c2 and c3 must be non-negative.
        # So, c1 ranges from 0 up to min(n, limit).
        for c1 in range(min(n, limit) + 1):
            
            # Calculate the remaining candies to be distributed between child 2 and child 3.
            remaining_candies_for_c2_c3 = n - c1
            
            # Iterate through possible number of candies for child 2 (c2)
            # c2 must be non-negative.
            # c2 cannot exceed 'limit'.
            # c2 cannot exceed 'remaining_candies_for_c2_c3', as c3 must be non-negative.
            # So, c2 ranges from 0 up to min(remaining_candies_for_c2_c3, limit).
            for c2 in range(min(remaining_candies_for_c2_c3, limit) + 1):
                
                # Candies for child 3 (c3) are determined by the remaining amount.
                c3 = n - c1 - c2
                
                # Check if c3 satisfies its constraints:
                # 1. c3 must be non-negative: This is guaranteed by the loop bounds of c1 and c2
                #    (specifically, c2 <= remaining_candies_for_c2_c3 implies c3 >= 0).
                # 2. c3 must not exceed 'limit'. This is the only explicit check needed.
                if c3 <= limit:
                    count += 1
        
        return count