class Solution:
    def maxSum(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def max_digit(num):
            return max(int(d) for d in str(num))
        
        max_digit_map = defaultdict(list)
        
        # Map each number to the maximum digit it contains
        for num in nums:
            max_digit_map[max_digit(num)].append(num)
        
        max_sum = -1
        
        # Iterate through each digit group and find the maximum sum of any two numbers
        for digit, values in max_digit_map.items():
            if len(values) > 1:
                # Sort the values to get the two largest easily
                values.sort()
                # Update max_sum with the sum of the two largest numbers in this group
                max_sum = max(max_sum, values[-1] + values[-2])
        
        return max_sum