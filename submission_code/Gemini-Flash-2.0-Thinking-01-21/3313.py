class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        coefficients = []
        for i in range(1, k + 1):
            coefficients.append((-1)**(i+1) * (k - i + 1))
        sorted_coefficients = sorted(coefficients, reverse=True)
        dp = {} # dp[i][j] is max strength using first i elements and selecting j subarrays
        
        def get_dp_value(index, count):
            if count == 0:
                return 0
            if index == 0:
                return -float('inf')
            if (index, count) in dp:
                return dp[(index, count)]
            
            # Option 1: don't use nums[index-1] in any subarray
            strength1 = get_dp_value(index - 1, count)
            
            # Option 2: end a subarray at index index-1, starting at some index l (0-indexed) from 0 to index-1
            max_strength2 = -float('inf')
            for start_index in range(index):
                current_subarray_sum = sum(nums[start_index:index])
                prev_strength = get_dp_value(start_index, count - 1)
                if prev_strength != -float('inf'):
                    strength2 = prev_strength + sorted_coefficients[count-1] * current_subarray_sum
                    max_strength2 = max(max_strength2, strength2)
                    
            result = max(strength1, max_strength2)
            dp[(index, count)] = result
            return result
            
        final_strength = get_dp_value(n, k)
        if final_strength == -float('inf'):
            return -1e18 # return very small value if no solution, but constraints suggest always possible. 
                         # For example 3, k=1, output is -1. 
        
        if final_strength == -float('inf'):
            max_single_element = max(nums) if nums else 0
            if k == 1:
                return max_single_element
            else:
                return 0 # No valid selection possible
                
        return final_strength