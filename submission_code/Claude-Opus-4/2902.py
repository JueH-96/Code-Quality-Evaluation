class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Helper function to find the maximum digit in a number
        def max_digit(n):
            max_d = 0
            while n > 0:
                max_d = max(max_d, n % 10)
                n //= 10
            return max_d
        
        # Group numbers by their maximum digit
        digit_groups = {}
        for num in nums:
            max_d = max_digit(num)
            if max_d not in digit_groups:
                digit_groups[max_d] = []
            digit_groups[max_d].append(num)
        
        max_sum = -1
        
        # For each group, find the maximum sum of any pair
        for group in digit_groups.values():
            if len(group) >= 2:
                # Sort in descending order to get the two largest numbers
                group.sort(reverse=True)
                # The sum of the two largest numbers will be the maximum for this group
                max_sum = max(max_sum, group[0] + group[1])
        
        return max_sum