class Solution:
    def minCostToEqualizeArray(self, nums: list, cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        sum_nums = sum(nums)
        min_num = min(nums)
        max_num = max(nums)
        
        min_cost = float('inf')
        # We'll iterate up to max_num + 1e6 to cover possible cases where higher T is better
        # This can be adjusted based on problem constraints and test cases
        upper_bound = max_num + 10**6
        for T in range(max_num, upper_bound + 1):
            sum_d = n * T - sum_nums
            if sum_d < 0:
                continue  # T is too low; all elements can't reach T
            
            max_d = T - min_num
            sum_d_minus_max = (n - 1) * T - (sum_nums - min_num)
            
            possible_two_ops = min(sum_d // 2, sum_d_minus_max)
            if possible_two_ops < 0:
                possible_two_ops = 0
            
            rem = sum_d - 2 * possible_two_ops
            if cost2 < 2 * cost1:
                current_cost = possible_two_ops * cost2 + rem * cost1
            else:
                current_cost = sum_d * cost1
            
            if current_cost < min_cost:
                min_cost = current_cost
        
        return min_cost % MOD