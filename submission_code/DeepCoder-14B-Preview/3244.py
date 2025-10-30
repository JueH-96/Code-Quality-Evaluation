import math

class Solution:
    def minimumArrayLength(self, nums: list[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        # Compute the GCD of the entire array
        current_gcd = nums[0]
        for num in nums[1:]:
            current_gcd = math.gcd(current_gcd, num)
            if current_gcd == 1:
                break  # GCD can't get lower than 1
        
        if current_gcd == 1:
            return 1
        else:
            # Check if all elements are the same
            all_same = True
            first = nums[0]
            for num in nums[1:]:
                if num != first:
                    all_same = False
                    break
            if all_same:
                # If all elements are the same, minimal length is 1 if even, else 2
                return 1 if n % 2 == 0 else 2
            else:
                # Else, minimal length is 2
                return 2