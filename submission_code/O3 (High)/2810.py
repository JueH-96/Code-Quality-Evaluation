from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        
        # best[i] – smallest price seen so far for chocolate type i
        best = nums[:]                           # after 0 rotations
        cur_sum = sum(best)                      # total chocolate price for 0 rotations
        answer = cur_sum                         # minimal total cost found so far (k = 0)
        
        # Try every possible number of rotations 1 … n-1
        for k in range(1, n):
            cur_sum = 0
            for i in range(n):
                # After k rotations, type i can also be bought from index (i - k) % n
                idx = (i - k) % n
                best[i] = min(best[i], nums[idx])
                cur_sum += best[i]
            
            # total cost = price of chocolates + cost of k rotations
            answer = min(answer, cur_sum + k * x)
        
        return answer