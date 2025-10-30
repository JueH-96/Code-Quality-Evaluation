class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        # Create a sorted list of (value, index) pairs
        sorted_nums = sorted((num, i) for i, num in enumerate(nums))
        
        # Initialize an index map to keep track of the current indices of the elements
        index_map = {i: i for i in range(len(nums))}
        
        # Iterate through the sorted list and try to place each element in its correct position
        for sorted_value, original_index in sorted_nums:
            # Find the current index of the element that should be placed here
            current_index = index_map[original_index]
            
            # Find the correct position where we can place this element
            for i in range(current_index - 1, -1, -1):
                # If the difference between the values is within the limit, swap them
                if abs(nums[i] - sorted_value) <= limit:
                    # Swap in the original array
                    nums[i], nums[current_index] = nums[current_index], nums[i]
                    
                    # Update the index map
                    index_map[original_index] = i
                    index_map[nums[current_index]] = current_index
                    
                    # Update the current index
                    current_index = i
                else:
                    # If we can't swap, break the loop
                    break
        
        return nums