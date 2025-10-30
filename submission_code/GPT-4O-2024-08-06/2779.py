class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        adjacent_count = 0
        
        for index, color in queries:
            # Check the current state of the index and its neighbors
            if index > 0 and nums[index] == nums[index - 1] and nums[index] != 0:
                adjacent_count -= 1
            if index < n - 1 and nums[index] == nums[index + 1] and nums[index] != 0:
                adjacent_count -= 1
            
            # Update the color
            nums[index] = color
            
            # Check the new state of the index and its neighbors
            if index > 0 and nums[index] == nums[index - 1]:
                adjacent_count += 1
            if index < n - 1 and nums[index] == nums[index + 1]:
                adjacent_count += 1
            
            # Append the current count of adjacent same-colored elements
            answer.append(adjacent_count)
        
        return answer