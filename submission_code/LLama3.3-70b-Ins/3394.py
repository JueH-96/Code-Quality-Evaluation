class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # Initialize the result array with the given x
        result = [x]
        
        # Since we need to ensure that each subsequent number is greater than the previous one,
        # we can simply add 1 to the previous number and check if the bitwise AND operation
        # with x is still x. If not, we add the smallest power of 2 that makes the bitwise AND
        # operation with x equal to x.
        for _ in range(n - 1):
            # Calculate the next number
            next_num = result[-1] + 1
            
            # While the bitwise AND operation between next_num and x is not x
            while (next_num & x) != x:
                # Add the smallest power of 2 that makes the bitwise AND operation with x equal to x
                next_num += 1
            
            # Append the next number to the result array
            result.append(next_num)
        
        # Return the last element of the result array
        return result[-1]