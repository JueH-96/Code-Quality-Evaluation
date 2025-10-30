class Solution:
    def maxSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Create a set to store unique elements
        unique_set = set()
        
        # Keep track of the maximum sum
        max_sum = 0
        
        # Iterate through the array
        for i in range(n):
            # If the current element is not in the set
            if nums[i] not in unique_set:
                # Add the element to the set
                unique_set.add(nums[i])
                
                # Calculate the current sum
                curr_sum = sum(unique_set)
                
                # Update the maximum sum if necessary
                max_sum = max(max_sum, curr_sum)
            
            # If the current element is in the set, remove it
            else:
                unique_set.remove(nums[i])
        
        return max_sum