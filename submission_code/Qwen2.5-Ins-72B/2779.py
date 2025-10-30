class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        count = 0
        
        for index, color in queries:
            # Check if the current index was part of a pair before coloring
            if nums[index] != 0:
                if index > 0 and nums[index - 1] == nums[index]:
                    count -= 1
                if index < n - 1 and nums[index + 1] == nums[index]:
                    count -= 1
            
            # Color the current index
            nums[index] = color
            
            # Check if the current index forms a new pair after coloring
            if index > 0 and nums[index - 1] == nums[index]:
                count += 1
            if index < n - 1 and nums[index + 1] == nums[index]:
                count += 1
            
            # Append the current count to the answer
            answer.append(count)
        
        return answer