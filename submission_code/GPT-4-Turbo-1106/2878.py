class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        # Calculate the prefix sum array
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + num)
        
        # Initialize the number of operations array
        operations = [0] * (len(nums) + 1)
        
        # Iterate through the array to simulate the operations
        for i in range(len(nums)):
            # Apply the operations accumulated so far
            prefix_sum[i] += operations[i]
            
            # Check if the current element can be reduced to zero
            if prefix_sum[i] < 0:
                # If the current prefix sum is negative, it's impossible to make all elements zero
                return False
            
            # Calculate the number of operations needed to make the current element zero
            operations_needed = prefix_sum[i]
            if operations_needed > 0:
                # If the subarray starting from the current element has less than k elements, return False
                if i + k > len(nums):
                    return False
                # Apply the operations to the current element and the next k-1 elements
                operations[i] -= operations_needed
                operations[i + k] += operations_needed
        
        # If the last k elements have been reduced to zero, return True
        return prefix_sum[-1] + operations[-1] == 0