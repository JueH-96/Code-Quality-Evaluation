class Solution:
    def minOperations(self, nums: List[int]) -> int:
        import sys
        input_data = sys.stdin.readline
        
        # -----------------------------------------------------------
        # Precompute the smallest prime factor (spf) for each number
        # up to 10^6 using a modified sieve approach.
        # spf[x] = the smallest prime factor of x (x >= 2).
        # For x=1, we'll define spf[x] = 1 for convenience (though 1 has no proper divisor).
        # -----------------------------------------------------------
        MAX_VAL = 10**6
        spf = [0]*(MAX_VAL+1)
        spf[1] = 1
        for i in range(2, MAX_VAL+1):
            if spf[i] == 0:  # i is prime if it hasn't been marked yet
                spf[i] = i
                # Mark multiples of i
                for multiple in range(i*i, MAX_VAL+1, i):
                    if spf[multiple] == 0:
                        spf[multiple] = i
        
        # -----------------------------------------------------------
        # For each number x in nums:
        #   - If x == 1 or x is prime (spf[x] == x), we have exactly 1 choice: x.
        #   - Otherwise (x is composite), we have 2 choices:
        #       1) Keep x (cost 0)
        #       2) Reduce x in one operation to x / (greatest proper divisor) = spf[x] (cost 1)
        #
        # We want to form a non-decreasing sequence from left to right
        # picking from these choices and minimizing total cost.
        #
        # We'll use dynamic programming with dp[i][j]:
        #   i = index in nums
        #   j = 0 or 1 (choosing the "0th + cost" or "1st + cost" option at index i)
        # We'll keep track of the minimal cost to achieve non-decreasing order
        # up to index i when choosing the j-th option at i.
        #
        # The j-th option's value:
        #   choice_value[i][0] = nums[i]
        #   choice_cost[i][0]  = 0
        #   choice_value[i][1] = spf[nums[i]]   (only if nums[i] is composite)
        #   choice_cost[i][1]  = 1             (only if nums[i] is composite)
        #
        # If nums[i] is prime or 1, then we do not actually have a second valid choice,
        # so dp[i][1] can be considered infinite unless spf[x] < x (composite case).
        # -----------------------------------------------------------

        n = len(nums)
        
        # Prepare the choice values and costs
        choice_val = [[0, 0] for _ in range(n)]
        choice_cost = [[0, 0] for _ in range(n)]
        
        for i in range(n):
            x = nums[i]
            choice_val[i][0] = x
            choice_cost[i][0] = 0
            # Check if x is composite: spf[x] < x for x>1
            if x > 1 and spf[x] < x:
                # second choice is spf[x]
                choice_val[i][1] = spf[x]
                choice_cost[i][1] = 1
            else:
                # prime or 1 => cannot reduce effectively
                choice_val[i][1] = float('inf')  # not a valid choice
                choice_cost[i][1] = float('inf')
        
        # dp[i][j] = minimal cost to achieve non-decreasing array up to i
        #            when we pick the j-th choice at index i (j in {0,1}).
        dp = [[float('inf'), float('inf')] for _ in range(n)]
        
        # Base case
        dp[0][0] = choice_cost[0][0]
        dp[0][1] = choice_cost[0][1]
        
        # Fill dp table
        for i in range(1, n):
            for j in range(2):  # j in {0,1} for i-th element
                val_j = choice_val[i][j]
                cost_j = choice_cost[i][j]
                if val_j == float('inf'):
                    # not a valid option
                    continue
                # try to match with dp[i-1][k]
                for k in range(2):  # k in {0,1} for (i-1)-th element
                    val_k = choice_val[i-1][k]
                    if dp[i-1][k] == float('inf'):
                        continue
                    # we need val_j >= val_k to keep non-decreasing
                    if val_j >= val_k:
                        dp[i][j] = min(dp[i][j], dp[i-1][k] + cost_j)
        
        # The answer is the minimum of dp[n-1][0] and dp[n-1][1]
        ans = min(dp[n-1][0], dp[n-1][1])
        return ans if ans != float('inf') else -1