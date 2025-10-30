class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        memo = {}
        
        def solve(index, current_sum, is_even_position, has_element):
            key = (index, current_sum, is_even_position, has_element)
            if key in memo:
                return memo[key]
            
            if index == len(nums):
                result = 1 if current_sum == k and has_element else -1
            else:
                # Option 1: Don't include nums[index]
                result1 = solve(index + 1, current_sum, is_even_position, has_element)
                
                # Option 2: Include nums[index]
                new_sum = current_sum + nums[index] if is_even_position else current_sum - nums[index]
                result2 = solve(index + 1, new_sum, not is_even_position, True)
                
                if result2 != -1:
                    new_product = result2 * nums[index]
                    result2 = new_product if new_product <= limit else -1
                
                result = max(result1, result2) if result1 != -1 and result2 != -1 else (result2 if result1 == -1 else result1)
            
            memo[key] = result
            return result
        
        answer = solve(0, 0, True, False)
        return answer if answer > 0 else -1