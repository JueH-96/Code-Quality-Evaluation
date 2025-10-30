class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        same_color_count = 0
        
        for index, color in queries:
            # Check the left neighbor
            if index > 0 and nums[index - 1] == color:
                if nums[index] != color:
                    same_color_count += 1
            elif index > 0 and nums[index] == nums[index - 1] and nums[index] != 0:
                same_color_count -= 1
            
            # Check the right neighbor
            if index < n - 1 and nums[index + 1] == color:
                if nums[index] != color:
                    same_color_count += 1
            elif index < n - 1 and nums[index] == nums[index + 1] and nums[index] != 0:
                same_color_count -= 1
            
            # Color the current index
            nums[index] = color
            
            # Append the current count to the answer list
            answer.append(same_color_count)
        
        return answer