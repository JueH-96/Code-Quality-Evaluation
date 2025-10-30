class Solution:
    def maxSum(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def max_digit(n):
            return max(int(d) for d in str(n))
        
        max_digit_map = defaultdict(list)
        
        for num in nums:
            digit = max_digit(num)
            max_digit_map[digit].append(num)
        
        max_sum = -1
        
        for digit, values in max_digit_map.items():
            if len(values) > 1:
                values.sort(reverse=True)
                max_sum = max(max_sum, values[0] + values[1])
        
        return max_sum