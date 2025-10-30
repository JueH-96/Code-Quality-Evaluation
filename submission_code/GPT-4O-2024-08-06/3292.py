class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Create a set to keep track of marked indices
        marked = set()
        
        # Create a list to keep track of the remaining decrements needed for each index
        remaining = nums[:]
        
        # Iterate over each second
        for second in range(m):
            # Get the current index to potentially mark
            current_index = changeIndices[second] - 1
            
            # If the current index is already marked, continue
            if current_index in marked:
                continue
            
            # If the current index can be marked (nums[current_index] == 0)
            if remaining[current_index] == 0:
                marked.add(current_index)
            else:
                # Otherwise, decrement the value at this index
                remaining[current_index] -= 1
            
            # Check if all indices are marked
            if len(marked) == n:
                return second + 1
        
        # If not all indices are marked by the end, return -1
        return -1