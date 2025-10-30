class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Dictionary to store numbers by their maximum digit.
        groups = {}
        
        # Function to extract max digit from a number
        def get_max_digit(n: int) -> int:
            max_d = 0
            while n > 0:
                d = n % 10
                if d > max_d:
                    max_d = d
                n //= 10
            return max_d
        
        # Populate groups by maximum digit
        for num in nums:
            max_digit = get_max_digit(num)
            if max_digit not in groups:
                groups[max_digit] = []
            groups[max_digit].append(num)
        
        max_sum = -1
        
        # For each group, if group size >= 2, consider two largest numbers
        for key in groups:
            if len(groups[key]) > 1:
                # Get two largest numbers. Using sort descending
                groups[key].sort(reverse=True)
                candidate_sum = groups[key][0] + groups[key][1]
                if candidate_sum > max_sum:
                    max_sum = candidate_sum
        
        return max_sum