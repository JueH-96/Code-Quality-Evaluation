from typing import List

class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + (num % modulo == k))
        
        count = 0
        mod_count = [0] * modulo
        mod_count[0] = 1
        
        for ps in prefix_sum:
            mod = ps % modulo
            target = (mod - k + modulo) % modulo
            count += mod_count[target]
            mod_count[mod] += 1
        
        return count