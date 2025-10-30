class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        best_odd = -10**20
        best_even = -10**20
        X = 0
        
        for i in range(n):
            if i == 0:
                X = nums[0]
            elif i % 2 == 1:
                X -= nums[i]
            else:
                X += nums[i]
            
            if i == 0:
                dp_i = X
                best_even = dp_i + X
            else:
                candidates = [X]
                candidates.append(best_odd + X)
                candidates.append(best_even - X)
                dp_i = max(candidates)
                
                if i % 2 == 0:
                    if dp_i + X > best_even:
                        best_even = dp_i + X
                else:
                    if dp_i - X > best_odd:
                        best_odd = dp_i - X
        
        return dp_i