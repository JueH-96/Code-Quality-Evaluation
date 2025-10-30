import math

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        total = 0
        m = k - 2
        
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] >= nums[j]:
                    continue
                before_i = i
                after_j = (n - 1) - j
                current_contribution = 0
                
                max_a = min(before_i, m)
                for a in range(0, max_a + 1):
                    b = m - a
                    if b < 0 or b > after_j:
                        continue
                    try:
                        c_before = math.comb(before_i, a)
                    except:
                        c_before = 0
                    try:
                        c_after = math.comb(after_j, b)
                    except:
                        c_after = 0
                    current_contribution = (current_contribution + c_before * c_after) % MOD
                contribution = (diff := nums[j] - nums[i]) * current_contribution
                total = (total + contribution) % MOD
        
        return total % MOD