class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 1:
            return 0
        
        min_num = min(nums)
        max_num = max(nums)
        sum_nums = sum(nums)
        
        # Check if using pair operations is not beneficial
        if cost2 >= 2 * cost1:
            sum_d = n * max_num - sum_nums
            return (sum_d * cost1) % MOD
        
        # Define the cost computation function
        def compute_cost(T):
            sum_d = n * T - sum_nums
            if sum_d == 0:
                return 0
            max_d = T - min_num
            a = sum_d // 2
            b = (n - 1) * T - (sum_nums - min_num)
            x_max = min(a, b)
            x_max = max(x_max, 0)  # Ensure non-negative
            single_cost = cost1 * (sum_d - 2 * x_max)
            pair_cost = cost2 * x_max
            total = (single_cost + pair_cost) % MOD
            return total
        
        # Perform ternary search between max_num and a high value
        low = max_num
        high = max_num + (1 << 20)  # A large enough upper bound
        
        # Narrow down the search range
        while high - low > 3:
            mid1 = (2 * low + high) // 3
            mid2 = (low + 2 * high) // 3
            cost1_val = compute_cost(mid1)
            cost2_val = compute_cost(mid2)
            if cost1_val < cost2_val:
                high = mid2
            else:
                low = mid1
        
        # Check all values in the narrowed range to find the minimum cost
        min_cost = compute_cost(low)
        for T in range(low + 1, high + 1):
            current_cost = compute_cost(T)
            if current_cost < min_cost:
                min_cost = current_cost
        
        return min_cost % MOD