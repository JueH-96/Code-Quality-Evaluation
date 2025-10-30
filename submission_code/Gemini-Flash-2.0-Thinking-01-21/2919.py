class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        n = len(usageLimits)
        current_usage_limits = list(usageLimits)
        num_groups = 0
        group_length = 1
        while True:
            available_numbers = []
            for i in range(n):
                if current_usage_limits[i] > 0:
                    available_numbers.append(i)
            if len(available_numbers) < group_length:
                break
            
            number_limit_pairs = []
            for index in available_numbers:
                number_limit_pairs.append((current_usage_limits[index], index))
            
            number_limit_pairs.sort(key=lambda x: x[0], reverse=True)
            
            indices_to_use = []
            for i in range(group_length):
                indices_to_use.append(number_limit_pairs[i][1])
                
            for index in indices_to_use:
                current_usage_limits[index] -= 1
                
            num_groups += 1
            group_length += 1
            
        return num_groups