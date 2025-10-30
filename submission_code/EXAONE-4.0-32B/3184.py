class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        a = [nums[i] - i for i in range(n)]
        
        sorted_a = sorted(set(a))
        comp = {val: idx for idx, val in enumerate(sorted_a)}
        m = len(sorted_a)
        
        NEG_INF = -10**18
        fenw = [NEG_INF] * (m + 1)
        
        def update(pos, val):
            idx = pos + 1
            while idx <= m:
                if val > fenw[idx]:
                    fenw[idx] = val
                idx += idx & -idx
        
        def query(pos):
            idx = pos + 1
            res = NEG_INF
            while idx:
                if fenw[idx] > res:
                    res = fenw[idx]
                idx -= idx & -idx
            return res
        
        ans = NEG_INF
        for i in range(n):
            c = comp[a[i]]
            pre_max = query(c)
            add = pre_max if pre_max > 0 else 0
            current_sum = nums[i] + add
            if current_sum > ans:
                ans = current_sum
            update(c, current_sum)
        
        return ans