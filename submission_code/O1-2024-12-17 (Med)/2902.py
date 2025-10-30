class Solution:
    def maxSum(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        # Dictionary to store the two largest numbers for each maximum digit
        max_digit_map = defaultdict(lambda: [0, 0])  # will hold [largest, second_largest]
        
        def get_max_digit(x: int) -> int:
            return max(int(d) for d in str(x))
        
        # Populate the dictionary
        for num in nums:
            d = get_max_digit(num)
            # Update the two largest values for the digit d
            if num > max_digit_map[d][0]:
                max_digit_map[d][1] = max_digit_map[d][0]
                max_digit_map[d][0] = num
            elif num > max_digit_map[d][1]:
                max_digit_map[d][1] = num
        
        # Compute the maximum sum among digits with at least two numbers
        max_sum = -1
        for d in max_digit_map:
            if max_digit_map[d][1] > 0:  # means we have at least two numbers
                s = max_digit_map[d][0] + max_digit_map[d][1]
                max_sum = max(max_sum, s)
        
        return max_sum