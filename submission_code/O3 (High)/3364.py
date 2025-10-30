from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        INF = 10 ** 18                          # a value bigger than any possible answer
        
        # dp_prev[i] : minimal sum for the first i elements split into
        #              (already finished) k sub-arrays
        dp_prev = [INF] * (n + 1)
        dp_prev[0] = 0                          # zero elements – zero cost for k = 0
        
        for k in range(m):                      # build the k-th sub-array (0-based)
            target = andValues[k]
            dp_curr = [INF] * (n + 1)
            
            # prev_ands keeps, for the previous index (j-1), all different
            # AND results of sub-arrays that END at j-1 together with the
            # minimal cost of the prefix **before** those sub-arrays started.
            prev_ands = {}
            
            for j in range(1, n + 1):           # j = length of prefix we consider
                new_ands = {}
                
                # start a new sub-array consisting only of nums[j-1]
                if dp_prev[j - 1] < INF:
                    val = nums[j - 1]
                    new_ands[val] = dp_prev[j - 1]
                
                # extend all sub-arrays that ended at position j-2
                for and_val, cost in prev_ands.items():
                    new_and = and_val & nums[j - 1]
                    # keep the minimal prefix cost that leads to this new_and
                    if cost < new_ands.get(new_and, INF):
                        new_ands[new_and] = cost
                        
                # if we can end the k-th sub-array here with the required AND
                if target in new_ands:
                    dp_curr[j] = new_ands[target] + nums[j - 1]   # add value of the sub-array
                
                prev_ands = new_ands            # move window
            dp_prev = dp_curr                   # next segment takes this as prefix table
        
        result = dp_prev[n]
        return -1 if result >= INF else result