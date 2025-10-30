class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Function to count the number of 1s in the binary representation of a number
        def count_ones(x):
            return bin(x).count('1')
        
        # Calculate the initial XOR of all elements in nums
        current_xor = 0
        for num in nums:
            current_xor ^= num
        
        # If the current XOR is already equal to k, no operations are needed
        if current_xor == k:
            return 0
        
        # Calculate the target XOR we need to achieve
        target_xor = current_xor ^ k
        
        # Dictionary to store the minimum number of operations to achieve a certain XOR
        dp = defaultdict(lambda: float('inf'))
        dp[0] = 0
        
        # Iterate over each number in nums
        for num in nums:
            # Create a copy of the current state of dp
            new_dp = dp.copy()
            for xor_val in dp:
                new_xor = xor_val ^ num
                new_dp[new_xor] = min(new_dp[new_xor], dp[xor_val] + count_ones(num))
            dp = new_dp
        
        # Return the minimum number of operations to achieve the target XOR
        return dp[target_xor]