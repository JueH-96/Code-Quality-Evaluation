class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        operations = 0  # Current number of active operations
        end_operations = [0] * n  # Track when operations end
        
        for i in range(n):
            # Stop operations that end at position i
            operations -= end_operations[i]
            
            # Calculate current value after all active operations
            current_value = nums[i] - operations
            
            # If value is negative, impossible to fix
            if current_value < 0:
                return False
            
            # If value is positive, must start operations here
            if current_value > 0:
                # Check if we can start k-length subarray at position i
                if i + k > n:
                    return False
                
                # Start current_value operations at position i
                operations += current_value
                
                # These operations will end after position i+k-1
                # So they stop affecting elements at position i+k
                if i + k < n:
                    end_operations[i + k] += current_value
        
        return True