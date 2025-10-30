from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        max_num = 10**6
        spf = list(range(max_num + 1))
        for i in range(2, int(max_num**0.5) + 1):
            if spf[i] == i:
                for j in range(i * i, max_num + 1, i):
                    if spf[j] == j:
                        spf[j] = i

        def get_pairs(x):
            if x == 1:
                return [(1, 0)]
            p = spf[x]
            if p == x:
                return [(x, 0)]
            else:
                return [(x, 0), (p, 1)]
        
        if not nums:
            return 0
        
        previous_dp = get_pairs(nums[0])
        if not previous_dp:
            return -1
        
        for num in nums[1:]:
            current_possible = get_pairs(num)
            new_dp = {}
            for v, s in current_possible:
                min_prev = float('inf')
                for pv, ps in previous_dp:
                    if pv <= v and ps < min_prev:
                        min_prev = ps
                if min_prev != float('inf'):
                    total = min_prev + s
                    if v not in new_dp or total < new_dp[v]:
                        new_dp[v] = total
            if not new_dp:
                return -1
            previous_dp = sorted(new_dp.items(), key=lambda x: x[0])
        
        return min(previous_dp, key=lambda x: x[1])[1]