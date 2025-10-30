class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        count = 0
        result = []
        
        for index, color in queries:
            # Calculate old contribution from pairs involving this index
            old_contribution = 0
            if index > 0 and nums[index-1] == nums[index] and nums[index] != 0:
                old_contribution += 1
            if index < n-1 and nums[index] == nums[index+1] and nums[index] != 0:
                old_contribution += 1
            
            # Update the array
            nums[index] = color
            
            # Calculate new contribution from pairs involving this index
            new_contribution = 0
            if index > 0 and nums[index-1] == nums[index] and nums[index] != 0:
                new_contribution += 1
            if index < n-1 and nums[index] == nums[index+1] and nums[index] != 0:
                new_contribution += 1
            
            # Update the count
            count = count - old_contribution + new_contribution
            result.append(count)
        
        return result