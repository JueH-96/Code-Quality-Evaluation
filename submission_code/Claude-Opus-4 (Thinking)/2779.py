class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        count = 0
        
        for index, color in queries:
            # Check and remove existing adjacent matches before changing
            if index > 0 and nums[index - 1] != 0 and nums[index - 1] == nums[index]:
                count -= 1
            if index < n - 1 and nums[index] != 0 and nums[index] == nums[index + 1]:
                count -= 1
            
            # Apply the color change
            nums[index] = color
            
            # Check and add new adjacent matches after changing
            if index > 0 and nums[index - 1] != 0 and nums[index - 1] == nums[index]:
                count += 1
            if index < n - 1 and nums[index] != 0 and nums[index] == nums[index + 1]:
                count += 1
            
            answer.append(count)
        
        return answer