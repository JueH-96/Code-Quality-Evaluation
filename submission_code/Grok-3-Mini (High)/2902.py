class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def max_digit(num):
            return max(int(d) for d in str(num))
        
        # Group numbers by their maximum digit
        groups = [[] for _ in range(10)]  # Index 0 is unused, 1-9 for max digits
        
        for num in nums:
            md = max_digit(num)
            groups[md].append(num)
        
        # Variable to keep track of the maximum sum found
        max_sum = -1
        
        # Check each group for digits 1 to 9
        for digit in range(1, 10):
            group = groups[digit]
            if len(group) >= 2:
                # Sort the group in descending order and sum the two largest
                group.sort(reverse=True)
                current_sum = group[0] + group[1]
                # Update max_sum if current_sum is larger
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum