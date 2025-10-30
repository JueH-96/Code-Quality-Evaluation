class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_groups = {}
        for num in nums:
            m = self.get_max_digit(num)
            if m not in max_groups:
                max_groups[m] = []
            max_groups[m].append(num)
        
        max_sum = -1
        for group in max_groups.values():
            if len(group) >= 2:
                group_sorted = sorted(group, reverse=True)
                current_sum = group_sorted[0] + group_sorted[1]
                if current_sum > max_sum:
                    max_sum = current_sum
        return max_sum
    
    def get_max_digit(self, num):
        max_d = 0
        while num > 0:
            d = num % 10
            if d > max_d:
                max_d = d
            num = num // 10
        return max_d