from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        groups = defaultdict(list)
        for num in nums:
            max_digit = max(map(int, str(num)))
            groups[max_digit].append(num)
        
        max_sum = -1
        for key in groups:
            if len(groups[key]) >= 2:
                groups[key].sort(reverse=True)
                current_sum = groups[key][0] + groups[key][1]
                if current_sum > max_sum:
                    max_sum = current_sum
        return max_sum