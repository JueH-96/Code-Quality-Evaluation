class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        
        # dp[i] will hold the minimum cost found so far to buy the chocolate of type i,
        # given we can make up to s shifts (for some s). We will iterate s from 0 to n,
        # updating dp as we go and computing the total possible cost each time.
        dp = nums[:]  # For s = 0 (no shifts), the cost to buy type i is just nums[i].
        best = sum(dp)  # Initial candidate: buy all chocolates without any shifts.
        
        # Try from 1 up to n shifts. Although shifting n times brings us back to
        # the original arrangement, intermediate states allow a type to appear at
        # different indices with possibly lower costs.
        running_sum = sum(dp)
        for s in range(1, n + 1):
            # Update dp for each type i:
            # dp[i] = min(dp[i], nums[(i - s) mod n])
            # but instead of doing a full nested loop each time, we do it step-by-step:
            new_sum = 0
            for i in range(n):
                # We'll update dp[i] by considering the chocolate if type i
                # were bought after s shifts (which places type i at index (i - s) mod n).
                dp[i] = min(dp[i], nums[(i - s) % n])
                new_sum += dp[i]
            
            running_sum = new_sum
            candidate = running_sum + s * x
            best = min(best, candidate)
        
        return best