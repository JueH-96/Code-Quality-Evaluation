class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        seconds = 0
        
        while True:
            # Check if all elements are equal
            if len(set(nums)) == 1:
                return seconds
            
            # Create a new list to store the updated values
            new_nums = [0] * n
            
            # Update the values in the new list
            for i in range(n):
                new_nums[i] = min(nums[i], nums[(i - 1 + n) % n], nums[(i + 1) % n])
            
            # Update the original list with the new values
            nums = new_nums
            
            # Increment the number of seconds
            seconds += 1