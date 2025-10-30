from typing import List

MOD = 10**9 + 7

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        """
        For every non empty subset we need   max(nums)^2 * min(nums)
        Sort the array. 
        While scanning the sorted array we treat the current element a[i] as the
        (unique) maximum of the subsets that end at position i.
        
        For those subsets two situations occur
          1) subset == {a[i]}                               ->  a[i]^3
          2) subset  = {a[i]} ∪ T  ,  T ⊆ {a[0]..a[i-1]}, T≠∅
             their power is  a[i]^2 * min(T)
        
        So for the current i we need  
             contribution = a[i]^3 + a[i]^2 * (sum over all non-empty subsets
                                                of the prefix {a[0]..a[i-1]}
                                                of their minimum)
        Let   prefixMinSum   be that latter sum for the current prefix.
        It can be updated in O(1):
             prefixMinSum_(new) = 2 * prefixMinSum_(old) + a[i]
        because every old subset can be kept as is or unioned with a[i] and
        a[i] itself forms one extra subset.
        
        Time  :  O(n log n) (for the sort)  
        Memory:  O(1)
        """
        nums.sort()
        
        prefix_min_sum = 0    # sum of the minima of ALL non-empty subsets of the already processed prefix
        ans = 0
        
        for x in nums:
            x_mod = x % MOD
            x2 = (x_mod * x_mod) % MOD        # x^2 mod MOD
            x3 = (x2 * x_mod) % MOD           # x^3 mod MOD
            
            # contribution when x is the maximum
            contribution = (x3 + x2 * prefix_min_sum) % MOD
            ans = (ans + contribution) % MOD
            
            # update prefix_min_sum for the next iteration
            prefix_min_sum = (prefix_min_sum * 2 + x_mod) % MOD
            
        return ans