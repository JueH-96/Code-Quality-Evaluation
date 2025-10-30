class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        max_num = 50
        previous = [0] * (max_num + 1)
        # Initialize for the first element
        first_num = nums[0]
        for v in range(first_num + 1):
            previous[v] = 1
        
        for i in range(1, n):
            current = [0] * (max_num + 1)
            delta = max(0, nums[i] - nums[i-1])
            for u in range(max_num + 1):
                if previous[u] == 0:
                    continue
                min_v = u + delta
                max_v = nums[i]
                if min_v > max_v:
                    continue
                # Update all v in [min_v, max_v]
                for v in range(min_v, max_v + 1):
                    current[v] = (current[v] + previous[u]) % MOD
            previous = current
        
        return sum(previous) % MOD