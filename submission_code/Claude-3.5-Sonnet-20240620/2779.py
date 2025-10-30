class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        count = 0
        
        for index, color in queries:
            old_color = nums[index]
            nums[index] = color
            
            if index > 0:
                if old_color != 0 and old_color == nums[index-1]:
                    count -= 1
                if color != 0 and color == nums[index-1]:
                    count += 1
            
            if index < n - 1:
                if old_color != 0 and old_color == nums[index+1]:
                    count -= 1
                if color != 0 and color == nums[index+1]:
                    count += 1
            
            answer.append(count)
        
        return answer