class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        count = 0
        result = []
        
        for index, color in queries:
            old_color = nums[index]
            
            # If the old and new colors are the same, the count doesn't change
            if old_color == color:
                result.append(count)
                continue
            
            # Check if we need to decrement count for existing pairs
            if index > 0 and old_color != 0 and nums[index - 1] == old_color:
                count -= 1
            
            if index < n - 1 and old_color != 0 and nums[index + 1] == old_color:
                count -= 1
            
            # Update the color at the given index
            nums[index] = color
            
            # Check if we need to increment count for new pairs
            if index > 0 and nums[index - 1] == color and color != 0:
                count += 1
            
            if index < n - 1 and nums[index + 1] == color and color != 0:
                count += 1
            
            result.append(count)
        
        return result