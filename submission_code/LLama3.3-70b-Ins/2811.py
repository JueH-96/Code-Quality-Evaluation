class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # Initialize the sum and the current number
        total_sum = 0
        curr_num = 1
        
        # Loop through the range of n
        for _ in range(n):
            # If the current number and k - current number are not the same and 
            # k - current number is not already in the sum, add the current number to the sum
            if k - curr_num != curr_num and k - curr_num not in [i for i in range(1, curr_num)]:
                total_sum += curr_num
            # If the current number and k - current number are the same or 
            # k - current number is already in the sum, add the next number to the sum
            else:
                curr_num += 1
                total_sum += curr_num
            # Increment the current number
            curr_num += 1
        
        return total_sum