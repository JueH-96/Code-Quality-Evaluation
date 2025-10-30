class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        count = 0
        
        # Iterate through all possible candies for child 1
        for child1 in range(min(n, limit) + 1):
            # Iterate through all possible candies for child 2
            for child2 in range(min(n - child1, limit) + 1):
                # Calculate candies for child 3
                child3 = n - child1 - child2
                
                # Check if child 3's allocation is valid
                if 0 <= child3 <= limit:
                    count += 1
        
        return count