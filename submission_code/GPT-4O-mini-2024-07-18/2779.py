class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        adjacent_count = 0
        
        for index, color in queries:
            # Check the left neighbor
            if index > 0 and nums[index] != 0 and nums[index] == nums[index - 1]:
                adjacent_count -= 1
            if index > 0 and color != 0 and nums[index - 1] == color:
                adjacent_count += 1
            
            # Check the right neighbor
            if index < n - 1 and nums[index] != 0 and nums[index] == nums[index + 1]:
                adjacent_count -= 1
            if index < n - 1 and color != 0 and nums[index + 1] == color:
                adjacent_count += 1
            
            # Color the current index
            nums[index] = color
            
            # Check again after coloring
            if index > 0 and nums[index] != 0 and nums[index] == nums[index - 1]:
                adjacent_count += 1
            if index < n - 1 and nums[index] != 0 and nums[index] == nums[index + 1]:
                adjacent_count += 1
            
            answer.append(adjacent_count)
        
        return answer