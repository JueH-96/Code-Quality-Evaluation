class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        # Initialize the answer list with zeros
        answer = [0] * len(queries)
        
        # Create a list to keep track of marked elements
        marked = [False] * len(nums)
        
        # Sort the indices of nums based on their values, then by index
        sorted_indices = sorted(range(len(nums)), key=lambda x: (nums[x], x))
        
        # Initialize a pointer to the first unmarked element in the sorted list
        unmarked_pointer = 0
        
        # Process each query
        for i, (index, k) in enumerate(queries):
            # Mark the element at the given index if it's not already marked
            if not marked[index]:
                marked[index] = True
                k -= 1  # Decrease k since we've marked one element
            
            # Mark k unmarked elements with the smallest values
            while k > 0 and unmarked_pointer < len(nums):
                # Find the next unmarked element in the sorted list
                while unmarked_pointer < len(nums) and marked[sorted_indices[unmarked_pointer]]:
                    unmarked_pointer += 1
                
                # If there is an unmarked element, mark it
                if unmarked_pointer < len(nums):
                    marked[sorted_indices[unmarked_pointer]] = True
                    unmarked_pointer += 1
                    k -= 1
            
            # Calculate the sum of unmarked elements
            answer[i] = sum(nums[j] for j in range(len(nums)) if not marked[j])
        
        return answer