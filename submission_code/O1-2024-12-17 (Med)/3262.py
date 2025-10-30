class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the array in non-decreasing order
        nums.sort()
        
        # Compute the total sum of all elements
        total_sum = sum(nums)
        i = len(nums) - 1
        
        # We will try removing the largest element while the polygon condition is not met
        # and keep checking with the new largest element in the reduced array.
        while i >= 2:
            # The largest element at index i
            largest = nums[i]
            other_sum = total_sum - largest
            
            # Check the polygon condition: sum of the other sides must be greater than the largest side
            if other_sum > largest:
                # We can form the polygon, so return its perimeter (the total sum)
                return total_sum
            
            # Otherwise, remove this largest element from consideration and continue
            total_sum = other_sum
            i -= 1
        
        # If no valid polygon of at least 3 sides can be formed, return -1
        return -1