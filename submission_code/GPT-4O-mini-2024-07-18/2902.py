class Solution:
    def maxSum(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def max_digit(n):
            return max(int(d) for d in str(n))
        
        digit_map = defaultdict(list)
        
        for num in nums:
            digit = max_digit(num)
            digit_map[digit].append(num)
        
        max_sum = -1
        
        for num_list in digit_map.values():
            if len(num_list) > 1:
                num_list.sort(reverse=True)
                max_sum = max(max_sum, num_list[0] + num_list[1])
        
        return max_sum