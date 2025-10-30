from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        nums.sort()
        
        ans = 0
        # T at step j will hold sum_{i<j} nums[i] * 2^{j-i-1} mod
        T = 0
        
        for x in nums:
            # Contribution from the subset {x} alone: x^3
            x2 = x * x % mod
            ans = (ans + x2 * x) % mod
            
            # Contribution from all subsets where x is the maximum
            # and some smaller element is the minimum:
            ans = (ans + x2 * T) % mod
            
            # Update T for the next iteration:
            # T_new = sum_{i<j+1} nums[i] * 2^{(j+1)-i-1}
            #       = sum_{i<j} nums[i] * 2^{j-i} + nums[j] * 2^0
            #       = 2*T + nums[j]
            T = (2 * T + x) % mod
        
        return ans