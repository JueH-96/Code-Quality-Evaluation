class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # Keep track of how many operations are currently active
        active_ops = 0
        # Keep track of when operations end
        ops_ending = [0] * n
        
        for i in range(n):
            # Remove operations that end at this position
            active_ops -= ops_ending[i]
            
            # Current effective value after applying active operations
            current_val = nums[i] - active_ops
            
            if current_val < 0:
                return False
            
            if current_val > 0:
                # We need to start current_val operations at position i
                needed_ops = current_val
                
                # Check if we can start operations at position i
                # (we need at least k elements from position i)
                if i + k > n:
                    return False
                
                # Start the operations
                active_ops += needed_ops
                
                # These operations will end after position i + k - 1
                if i + k < n:
                    ops_ending[i + k] += needed_ops
        
        return True