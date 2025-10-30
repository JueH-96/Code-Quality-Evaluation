class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        count = 0
        
        for index, color in queries:
            # If changing an existing color, check if we need to decrement count
            if nums[index] != 0:
                # Check left neighbor
                if index > 0 and nums[index-1] == nums[index]:
                    count -= 1
                # Check right neighbor    
                if index < n-1 and nums[index+1] == nums[index]:
                    count -= 1
            
            # Set new color
            nums[index] = color
            
            # Check if new color creates matches with neighbors
            if index > 0 and nums[index-1] == color:
                count += 1
            if index < n-1 and nums[index+1] == color:
                count += 1
                
            answer.append(count)
            
        return answer