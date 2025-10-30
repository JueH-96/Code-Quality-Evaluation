class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10 ** 9 + 7
        
        # If n == 1, the minimum possible sum is 1
        if n == 1:
            return 1
        
        # If target == 1, the minimum possible sum is n
        if target == 1:
            return n
        
        # Initialize the minimum possible sum to n * (n + 1) // 2, which is the sum of the first n positive integers
        min_sum = n * (n + 1) // 2
        
        # Iterate through all possible values for the first element of the array
        for first_num in range(1, min_sum // (n - 1) + 1):
            # Calculate the remaining sum needed for the other n-1 elements
            remaining_sum = min_sum - first_num
            
            # If the remaining sum is divisible by n-1 and the quotient is distinct from the first element,
            # then we have found a valid beautiful array
            if remaining_sum % (n - 1) == 0 and remaining_sum // (n - 1) != first_num:
                return (first_num + remaining_sum // (n - 1)) % MOD
        
        # If no valid beautiful array is found, return the minimum possible sum
        return min_sum % MOD