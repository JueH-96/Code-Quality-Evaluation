class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # The minimum possible value of nums[n - 1] is determined by the requirement
        # that nums[i + 1] > nums[i] and the AND of all elements equals x.
        
        # If n is 1, the only element can be x itself.
        if n == 1:
            return x
        
        # The minimum last element can be calculated as follows:
        # We need to ensure that the last element is greater than x and also
        # that the AND of all elements equals x.
        
        # Start with the last element as x and increment it until we find a valid sequence.
        last_element = x
        while True:
            # Generate the array with the last element being last_element
            nums = [last_element - i for i in range(n)]
            nums.reverse()  # To ensure nums is increasing
            
            # Check if the first element is positive and the AND of all elements equals x
            if nums[0] > 0 and all(num > 0 for num in nums):
                and_result = nums[0]
                for num in nums[1:]:
                    and_result &= num
                if and_result == x:
                    return last_element
            last_element += 1