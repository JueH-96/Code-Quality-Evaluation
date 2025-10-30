class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        cost = []
        for i in range(n - x + 1):
            subarray = nums[i:i+x]
            subarray_sorted = sorted(subarray)
            median = subarray_sorted[x // 2]
            total = sum(abs(num - median) for num in subarray)
            cost.append(total)
        
        m = len(cost)
        INF = float('inf')
        dp_prev = [INF] * (m + 1)
        dp_prev[0] = 0
        
        for j in range(1, k + 1):
            dp_current = [INF] * (m + 1)
            prefix_min = [INF] * (m + 1)
            prefix_min[0] = dp_prev[0]
            for i in range(1, m + 1):
                prefix_min[i] = min(prefix_min[i-1], dp_prev[i])
            for i in range(1, m + 1):
                # Option 1: not take i-th window
                option1 = dp_current[i-1]
                # Option 2: take i-th window
                max_i0 = i - x
                if max_i0 >= 0:
                    option2 = prefix_min[max_i0] + cost[i-1]
                else:
                    if j == 1:
                        option2 = cost[i-1]
                    else:
                        option2 = INF
                dp_current[i] = min(option1, option2)
            dp_prev = dp_current
        
        return dp_prev[m]