class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n
        
        for i in range(n):
            if nums[i] == 0:
                result[i] = nums[i]
            else:
                # Calculate steps and direction
                steps = abs(nums[i])
                # If positive, move right; if negative, move left
                direction = 1 if nums[i] > 0 else -1
                
                # Calculate final position considering circular nature
                final_pos = (i + direction * steps) % n
                
                # For negative indices, adjust to positive
                if final_pos < 0:
                    final_pos += n
                    
                # Store the value at the final position
                result[i] = nums[final_pos]
        
        return result