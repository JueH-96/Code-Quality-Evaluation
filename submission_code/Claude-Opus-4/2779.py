class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        count = 0
        result = []
        
        for index, color in queries:
            # Before coloring, check if we need to decrease count
            if index > 0 and nums[index] != 0 and nums[index] == nums[index - 1]:
                count -= 1
            if index < n - 1 and nums[index] != 0 and nums[index] == nums[index + 1]:
                count -= 1
            
            # Color the index
            nums[index] = color
            
            # After coloring, check if we need to increase count
            if index > 0 and nums[index] != 0 and nums[index] == nums[index - 1]:
                count += 1
            if index < n - 1 and nums[index] != 0 and nums[index] == nums[index + 1]:
                count += 1
            
            result.append(count)
        
        return result