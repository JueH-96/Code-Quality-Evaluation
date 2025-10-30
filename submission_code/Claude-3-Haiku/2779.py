class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        
        for index, color in queries:
            prev_color = nums[index]
            nums[index] = color
            
            count = 0
            if index > 0 and nums[index-1] == color:
                count += 1
            if index < n-1 and nums[index+1] == color:
                count += 1
            
            if prev_color != 0 and index > 0 and nums[index-1] == prev_color:
                count -= 1
            if prev_color != 0 and index < n-1 and nums[index+1] == prev_color:
                count -= 1
            
            answer.append(count)
        
        return answer