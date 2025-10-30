class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        result = []
        count = 0
        
        for index, color in queries:
            # Before changing, check current adjacent pairs involving this index
            old_count = 0
            
            # Check pair (index-1, index)
            if index > 0 and nums[index-1] != 0 and nums[index-1] == nums[index]:
                old_count += 1
            
            # Check pair (index, index+1)
            if index < n-1 and nums[index] != 0 and nums[index] == nums[index+1]:
                old_count += 1
            
            # Update the color
            nums[index] = color
            
            # After changing, check new adjacent pairs involving this index
            new_count = 0
            
            # Check pair (index-1, index)
            if index > 0 and nums[index-1] != 0 and nums[index-1] == nums[index]:
                new_count += 1
            
            # Check pair (index, index+1)
            if index < n-1 and nums[index] != 0 and nums[index] == nums[index+1]:
                new_count += 1
            
            # Update total count
            count = count - old_count + new_count
            result.append(count)
        
        return result