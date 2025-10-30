class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        count = 0
        answer = []
        
        for query in queries:
            index, color = query
            old_color = nums[index]
            nums[index] = color
            
            # Check pair (i-1, i)
            if index > 0:
                if nums[index-1] == old_color and old_color != 0:
                    if nums[index-1] != color:
                        count -= 1
                if nums[index-1] == color and color != 0:
                    if nums[index-1] != old_color:
                        count += 1
            
            # Check pair (i, i+1)
            if index < n-1:
                if nums[index+1] == old_color and old_color != 0:
                    if nums[index+1] != color:
                        count -= 1
                if nums[index+1] == color and color != 0:
                    if nums[index+1] != old_color:
                        count += 1
            
            answer.append(count)
        
        return answer