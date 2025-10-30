class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Find the longest suffix with distinct elements
        seen = set()
        start_of_distinct = n  # Start from the end
        
        # Traverse from right to left
        for i in range(n - 1, -1, -1):
            if nums[i] in seen:
                # Found a duplicate, so the distinct suffix starts after this
                start_of_distinct = i + 1
                break
            seen.add(nums[i])
        else:
            # All elements are distinct
            start_of_distinct = 0
        
        # Calculate number of operations needed
        # We need to remove all elements before start_of_distinct
        elements_to_remove = start_of_distinct
        
        # Each operation removes 3 elements
        operations = (elements_to_remove + 2) // 3  # Ceiling division
        
        return operations