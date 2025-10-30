class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Function to find maximum digit in a number
        def max_digit(n):
            return int(max(str(n)))
        
        # Group numbers by their maximum digit
        groups = {}
        for num in nums:
            md = max_digit(num)
            if md not in groups:
                groups[md] = []
            groups[md].append(num)
        
        # Find the maximum sum
        max_sum = -1
        for group in groups.values():
            if len(group) >= 2:
                # Sort the group in descending order
                group.sort(reverse=True)
                # The sum of the two largest numbers
                max_sum = max(max_sum, group[0] + group[1])
        
        return max_sum