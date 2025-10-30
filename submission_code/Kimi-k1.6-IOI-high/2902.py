class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Group numbers by their maximum digit
        groups = {}
        for num in nums:
            max_digit = max(int(d) for d in str(num))
            if max_digit not in groups:
                groups[max_digit] = []
            groups[max_digit].append(num)
        
        max_sum = -1
        # Check each group for possible pairs
        for group in groups.values():
            if len(group) >= 2:
                # Sort the group in descending order to get top two elements
                sorted_group = sorted(group, reverse=True)
                current_sum = sorted_group[0] + sorted_group[1]
                if current_sum > max_sum:
                    max_sum = current_sum
        return max_sum if max_sum != -1 else -1