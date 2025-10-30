class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def get_max_digit(num):
            return max(int(digit) for digit in str(num))
        
        # Group numbers by their maximum digit
        groups = {}
        for num in nums:
            max_digit = get_max_digit(num)
            if max_digit not in groups:
                groups[max_digit] = []
            groups[max_digit].append(num)
        
        max_sum = -1
        
        # For each group with at least 2 elements, find the maximum sum of any pair
        for group in groups.values():
            if len(group) >= 2:
                # Sort in descending order and take the two largest
                group.sort(reverse=True)
                current_sum = group[0] + group[1]
                max_sum = max(max_sum, current_sum)
        
        return max_sum