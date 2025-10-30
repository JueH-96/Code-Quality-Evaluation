class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        
        while nums:  # Continue until nums is empty
            # Sort nums to easily find min and max
            nums.sort()
            
            # Take the smallest and largest elements
            min_elem = nums[0]
            max_elem = nums[-1]
            
            # Remove these elements from nums
            nums = nums[1:-1]
            
            # Calculate their average and add it to averages
            avg = (min_elem + max_elem) / 2
            averages.append(avg)
        
        # Return the minimum element in averages
        return min(averages)