from typing import List
from collections import defaultdict

MOD = 10 ** 9 + 7

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        # count_end[v]   : number of good subsequences that currently end with value v
        # sum_end[v]     : total sum of elements over all those subsequences
        count_end = defaultdict(int)
        sum_end   = defaultdict(int)
        
        total_answer = 0
        
        for x in nums:
            # Sub-sequences that can be extended with x are those
            # whose last element is x-1 or x+1
            prev_cnt = (count_end.get(x - 1, 0) + count_end.get(x + 1, 0)) % MOD
            prev_sum = (sum_end  .get(x - 1, 0) + sum_end  .get(x + 1, 0)) % MOD
            
            # New subsequences formed by adding current x
            #   1) Append x to each extendable subsequence
            #   2) The single element subsequence [x]
            new_cnt = (prev_cnt + 1) % MOD
            new_sum = (prev_sum + prev_cnt * x + x) % MOD   # sums of all new subsequences
            
            # Update information for value x
            count_end[x] = (count_end[x] + new_cnt) % MOD
            sum_end[x]   = (sum_end[x]   + new_sum) % MOD
            
            # Accumulate to answer
            total_answer = (total_answer + new_sum) % MOD
        
        return total_answer