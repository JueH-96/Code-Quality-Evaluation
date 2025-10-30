class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        # Create a set to keep track of marked indices
        marked_indices = set()
        
        # Create a dictionary to keep track of the number of times each index needs to be decremented
        decrement_count = {i: nums[i-1] for i in range(1, len(nums) + 1)}
        
        # Iterate through each second
        for second, index in enumerate(changeIndices, start=1):
            # If the current index is already at 0, mark it
            if decrement_count[index] == 0:
                marked_indices.add(index)
            else:
                # Find an index to decrement
                for i in range(1, len(nums) + 1):
                    if decrement_count[i] > 0:
                        decrement_count[i] -= 1
                        break
            
            # Check if all indices are marked
            if len(marked_indices) == len(nums):
                return second
        
        # If not all indices are marked by the end, return -1
        return -1