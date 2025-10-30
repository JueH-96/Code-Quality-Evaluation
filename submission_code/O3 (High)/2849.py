from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        
        for i in range(n):
            # seen[k] tells whether value k has appeared in the current subarray
            # we need indices from 0 … n+1 (because we access x-1 and x+1)
            seen = [False] * (n + 2)
            seen[nums[i]] = True        # first element of the subarray
            imbalance = 0               # current imbalance for subarray nums[i … j]
            
            # extend the subarray to the right, one element at a time
            for j in range(i + 1, n):
                x = nums[j]
                if not seen[x]:
                    left  = seen[x - 1]   # is value x-1 already in the subarray?
                    right = seen[x + 1]   # is value x+1 already in the subarray?
                    
                    # update imbalance according to the appearance of gaps
                    if not left and not right:
                        # x does not connect to any neighbour ⇒ we create a new gap
                        imbalance += 1
                    elif left and right:
                        # x bridges an existing gap between (x-1) and (x+1)
                        imbalance -= 1
                        
                    seen[x] = True        # mark x as present
                    
                ans += imbalance          # add current imbalance of nums[i … j]
        
        return ans