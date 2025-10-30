class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        memo = {}
        
        def get_strength(start_index, subarrays_remaining):
            if subarrays_remaining == 0:
                return 0
            if start_index >= n:
                return -float('inf')
            if (start_index, subarrays_remaining) in memo:
                return memo[(start_index, subarrays_remaining)]
            
            result = get_strength(start_index + 1, subarrays_remaining) # Option 1: skip index start_index
            
            for end_index in range(start_index, n):
                current_subarray_sum = sum(nums[start_index:end_index+1])
                weight = ((-1) ** (subarrays_remaining + 1)) * (k - subarrays_remaining + 1)
                next_strength = get_strength(end_index + 1, subarrays_remaining - 1)
                if next_strength != -float('inf'):
                    value = current_subarray_sum * weight + next_strength
                    result = max(result, value)
                    
            memo[(start_index, subarrays_remaining)] = result
            return result
            
        max_strength_val = get_strength(0, k)
        return max_strength_val if max_strength_val != -float('inf') else 0