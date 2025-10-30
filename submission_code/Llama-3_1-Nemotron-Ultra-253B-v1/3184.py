import bisect
from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        dp = []
        for i, num in enumerate(nums):
            key_i = num - i
            # Find the largest key <= key_i
            pos = bisect.bisect_right(dp, (key_i, float('inf'))) - 1
            prev_max = dp[pos][1] if pos >= 0 else 0
            current_sum = max(prev_max + num, num)
            
            # Insert (key_i, current_sum) into dp
            insert_pos = bisect.bisect_left(dp, (key_i, -float('inf')))
            
            # Check if there's an entry with the same key
            if insert_pos < len(dp) and dp[insert_pos][0] == key_i:
                if dp[insert_pos][1] < current_sum:
                    dp[insert_pos] = (key_i, current_sum)
                else:
                    # No need to proceed
                    continue
            else:
                dp.insert(insert_pos, (key_i, current_sum))
            
            # Remove entries after insert_pos with sum <= current_sum
            j = insert_pos + 1
            while j < len(dp):
                if dp[j][1] <= current_sum:
                    del dp[j]
                else:
                    break
        
        return max([s for k, s in dp]) if dp else 0