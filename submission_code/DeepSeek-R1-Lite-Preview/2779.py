class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        count = 0
        answer = []
        
        for query in queries:
            index_i, color_i = query
            old_color = nums[index_i]
            
            # Check left neighbor before update
            if index_i - 1 >= 0:
                if old_color == nums[index_i - 1] and old_color != 0:
                    count -= 1
            # Check right neighbor before update
            if index_i + 1 < n:
                if old_color == nums[index_i + 1] and old_color != 0:
                    count -= 1
            
            # Update the color
            nums[index_i] = color_i
            
            # Check left neighbor after update
            if index_i - 1 >= 0:
                if color_i == nums[index_i - 1] and color_i != 0:
                    count += 1
            # Check right neighbor after update
            if index_i + 1 < n:
                if color_i == nums[index_i + 1] and color_i != 0:
                    count += 1
            
            # Append the current count to the answer
            answer.append(count)
        
        return answer