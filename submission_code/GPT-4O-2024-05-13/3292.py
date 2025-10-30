class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # Create a set to keep track of marked indices
        marked = set()
        
        # Create a list to keep track of remaining decrements needed for each index
        remaining = nums[:]
        
        # Iterate through each second
        for second in range(m):
            idx = changeIndices[second] - 1  # Convert to 0-indexed
            
            # Decrement the value at the current index if it's not already zero
            if remaining[idx] > 0:
                remaining[idx] -= 1
            
            # If the value at the current index is zero, mark the index
            if remaining[idx] == 0:
                marked.add(idx)
            
            # Check if all indices are marked
            if len(marked) == n:
                return second + 1  # Return the 1-indexed second
        
        # If not all indices are marked by the end, return -1
        return -1