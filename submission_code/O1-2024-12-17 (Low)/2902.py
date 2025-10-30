class Solution:
    def maxSum(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        # A dictionary where each key is the maximum digit of the numbers,
        # and the value is a list of numbers that have this maximum digit
        max_digit_map = defaultdict(list)
        
        for num in nums:
            # Find the maximum digit in the current number
            max_digit = max(int(d) for d in str(num))
            # Append the number to the corresponding list in the dictionary
            max_digit_map[max_digit].append(num)
        
        # Track the maximum sum
        max_sum_pair = -1
        
        # For each key in the dictionary, attempt to find the top two numbers
        for digit, numbers_list in max_digit_map.items():
            if len(numbers_list) > 1:
                # Sort descending and take the sum of the first two largest numbers
                numbers_list.sort(reverse=True)
                current_sum = numbers_list[0] + numbers_list[1]
                max_sum_pair = max(max_sum_pair, current_sum)
        
        return max_sum_pair