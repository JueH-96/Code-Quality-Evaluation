class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # Create a list of numbers from 1 to n
        nums = [i for i in range(1, n+1)]
        
        # Sort the list in ascending order
        nums.sort()
        
        # Initialize the sum to 0
        sum = 0
        
        # Loop through the list
        for i in range(n):
            # If the index is even, add the number at the even index to the sum
            if i % 2 == 0:
                sum += nums[i]
            # If the index is odd and the number of elements is even, subtract the number at the odd index from the sum
            elif i % 2 != 0 and n % 2 == 0:
                sum -= nums[i]
        
        # Return the sum
        return sum