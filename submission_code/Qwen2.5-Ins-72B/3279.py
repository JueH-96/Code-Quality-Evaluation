class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        # Initialize the count of valid pairs
        count = 0
        
        # Iterate over all possible values of x in the range [1, n]
        for x in range(1, n + 1):
            # Iterate over all possible values of y in the range [1, m]
            for y in range(1, m + 1):
                # Alice wins if the total number of flowers is odd
                if (x + y) % 2 == 1:
                    count += 1
        
        return count