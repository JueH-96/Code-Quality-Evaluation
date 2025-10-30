class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # Create a set to store distinct elements
        distinct_set = set()
        
        # Count the number of operations needed
        operations = 0
        
        while nums:
            # Remove the first 3 elements or all remaining elements if there are fewer than 3
            removed = nums[:3]
            nums = nums[3:]
            
            # Add the removed elements to the distinct set
            for num in removed:
                distinct_set.add(num)
            
            # Increment the number of operations
            operations += 1
            
            # If the length of the distinct set is equal to the length of the original array,
            # we have made all elements distinct, so we can return the number of operations
            if len(distinct_set) == len(nums) + len(removed):
                return operations
        
        # If the array is empty, all elements are distinct, so we return 0
        return 0