class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        answer = []
        current_pairs = 0
        
        for index, color in queries:
            if nums[index] == color:
                # If the color is the same as before, no change in pairs
                answer.append(current_pairs)
                continue
            
            # Check left and right neighbors
            left_same = (index > 0 and nums[index - 1] == nums[index] and nums[index] != 0)
            right_same = (index < n - 1 and nums[index + 1] == nums[index] and nums[index] != 0)
            
            # Update the current color
            nums[index] = color
            
            # Check new left and right conditions
            new_left_same = (index > 0 and nums[index - 1] == nums[index] and nums[index] != 0)
            new_right_same = (index < n - 1 and nums[index + 1] == nums[index] and nums[index] != 0)
            
            # Update the count of adjacent pairs
            if left_same:
                current_pairs -= 1
            if right_same:
                current_pairs -= 1
            if new_left_same:
                current_pairs += 1
            if new_right_same:
                current_pairs += 1
            
            # Append the result for this query
            answer.append(current_pairs)
        
        return answer