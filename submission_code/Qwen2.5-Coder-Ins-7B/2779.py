class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = [0] * len(queries)
        count = 0
        
        for i, (index, color) in enumerate(queries):
            if nums[index] != 0:
                if index > 0 and nums[index] == nums[index - 1]:
                    count -= 1
                if index < n - 1 and nums[index] == nums[index + 1]:
                    count -= 1
            
            nums[index] = color
            
            if index > 0 and nums[index] == nums[index - 1]:
                count += 1
            if index < n - 1 and nums[index] == nums[index + 1]:
                count += 1
            
            answer[i] = count
        
        return answer