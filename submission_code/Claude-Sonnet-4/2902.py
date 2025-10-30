class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def get_max_digit(num):
            return max(int(digit) for digit in str(num))
        
        # Group numbers by their maximum digit
        digit_groups = {}
        for num in nums:
            max_digit = get_max_digit(num)
            if max_digit not in digit_groups:
                digit_groups[max_digit] = []
            digit_groups[max_digit].append(num)
        
        max_sum = -1
        
        # For each group with at least 2 numbers, find the maximum pair sum
        for digit, group in digit_groups.items():
            if len(group) >= 2:
                # Sort in descending order to easily get the two largest numbers
                group.sort(reverse=True)
                # The maximum sum for this group is the sum of the two largest numbers
                current_sum = group[0] + group[1]
                max_sum = max(max_sum, current_sum)
        
        return max_sum