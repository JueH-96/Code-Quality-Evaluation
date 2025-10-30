class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = {}
        
        def solve(i, completed, in_subarray):
            # Base cases
            if completed == k:
                return 0
            if i == n:
                if in_subarray:
                    # If we're in a subarray at the end, we can complete it
                    return 0 if completed + 1 == k else -float('inf')
                else:
                    # Not in subarray and haven't completed k subarrays
                    return -float('inf')
            
            if (i, completed, in_subarray) in dp:
                return dp[(i, completed, in_subarray)]
            
            result = -float('inf')
            
            if in_subarray == 0:  # Not currently in a subarray
                # Option 1: Skip this element
                result = max(result, solve(i + 1, completed, 0))
                
                # Option 2: Start a new subarray with this element
                if completed < k:
                    # Coefficient for the (completed+1)-th subarray
                    coeff = (k - completed) if completed % 2 == 0 else -(k - completed)
                    result = max(result, nums[i] * coeff + solve(i + 1, completed, 1))
            
            else:  # Currently in a subarray
                # Option 1: Continue the subarray with this element
                coeff = (k - completed) if completed % 2 == 0 else -(k - completed)
                result = max(result, nums[i] * coeff + solve(i + 1, completed, 1))
                
                # Option 2: End the current subarray (not including this element)
                result = max(result, solve(i, completed + 1, 0))
            
            dp[(i, completed, in_subarray)] = result
            return result
        
        return solve(0, 0, 0)