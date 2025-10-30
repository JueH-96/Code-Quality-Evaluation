class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        from math import gcd
        from collections import Counter
        
        # Count the occurrences of each number
        count = Counter(nums)
        
        # The minimum length can be determined by the greatest common divisor (gcd)
        # of the numbers in the array. The gcd will determine the smallest number
        # that can be formed from the operations.
        unique_numbers = list(count.keys())
        
        if len(unique_numbers) == 1:
            return 1
        
        current_gcd = unique_numbers[0]
        for num in unique_numbers[1:]:
            current_gcd = gcd(current_gcd, num)
            if current_gcd == 1:  # If at any point gcd becomes 1, we can only reach 1
                return 1
        
        # The minimum length of the array will be the number of unique numbers
        # that can be formed by the gcd
        return len(unique_numbers) if current_gcd > 1 else 1