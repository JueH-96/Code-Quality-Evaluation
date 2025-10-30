from typing import List

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        if not nums:
            return 0
        
        # The maximum possible value in nums
        max_val = max(nums)
        # Counters and sum accumulators for subsequences ending with value v
        cnt = [0] * (max_val + 3)
        sumv = [0] * (max_val + 3)
        
        ans = 0
        
        for a in nums:
            # counts and sums of subsequences ending with a-1 and a+1
            c1 = cnt[a - 1] if a - 1 >= 0 else 0
            s1 = sumv[a - 1] if a - 1 >= 0 else 0
            c2 = cnt[a + 1]
            s2 = sumv[a + 1]
            
            # number of new subsequences ending at a:
            #   the singleton [a], plus extensions of those ending with a-1 or a+1
            new_cnt = (1 + c1 + c2) % MOD
            
            # total sum of elements of these new subsequences:
            #   a for the singleton, plus for each extension we add its old sum
            #   and also add 'a' to each of them => c1*a + c2*a
            new_sumv = (a + (s1 + c1 * a) + (s2 + c2 * a)) % MOD
            
            # update accumulators
            cnt[a] = (cnt[a] + new_cnt) % MOD
            sumv[a] = (sumv[a] + new_sumv) % MOD
            
            ans = (ans + new_sumv) % MOD
        
        return ans