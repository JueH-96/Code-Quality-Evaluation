class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # We want to choose at most op1 indices to halve once (rounding up),
        # and at most op2 indices to subtract k once (only if >= k).
        # We can also do both operations on the same index if allowed.
        #
        # We'll use DP to maximize the total "savings" from applying operations.
        # For each element, we consider up to four possibilities:
        #   1) No operations.
        #   2) Only halve (if we still can use an op1 "slot").
        #   3) Only subtract k (if the element >= k and we still can use an op2 "slot").
        #   4) Both halve and subtract k (if element >= k and we still have both slots).
        #
        # Let saving_0[i] = 0 (no operation)
        # Let saving_1[i] = nums[i] - ceil(nums[i]/2)  (apply op1 only, if allowed)
        # Let saving_2[i] = nums[i] - (nums[i]-k) = k, if nums[i] >= k, else 0
        # Let saving_12[i] = nums[i] - ceil((nums[i]-k)/2), if nums[i] >= k, else 0
        #
        # We'll define dp[i][c1][c2] = max total savings using first i elements
        # with c1 uses of op1 left and c2 uses of op2 left.
        #
        # The final answer will be sum(nums) - dp[n][op1][op2].

        n = len(nums)
        # Precompute the savings for each element
        saving_1 = [0]*n
        saving_2 = [0]*n
        saving_12 = [0]*n
        
        for i in range(n):
            saving_1[i] = nums[i] - ((nums[i] + 1)//2)  # ceil division by 2
            if nums[i] >= k:
                saving_2[i] = k
                # subtract k, then halve (round up)
                reduced = nums[i] - k
                saving_12[i] = nums[i] - ((reduced + 1)//2)

        # dp array dimensions: (n+1) x (op1+1) x (op2+1)
        dp = [[[0]*(op2+1) for _ in range(op1+1)] for _ in range(n+1)]
        
        for i in range(1, n+1):
            val_1 = saving_1[i-1]
            val_2 = saving_2[i-1]
            val_12 = saving_12[i-1]
            
            for c1 in range(op1+1):
                for c2 in range(op2+1):
                    # Option 0: no ops on i-th element
                    dp[i][c1][c2] = max(dp[i][c1][c2], dp[i-1][c1][c2])
                    
                    # Option 1: only op1 (if c1>0)
                    if c1 > 0:
                        candidate = dp[i-1][c1-1][c2] + val_1
                        dp[i][c1][c2] = max(dp[i][c1][c2], candidate)
                    
                    # Option 2: only op2 (if c2>0 and nums[i-1]>=k)
                    if c2 > 0 and saving_2[i-1] > 0:
                        candidate = dp[i-1][c1][c2-1] + val_2
                        dp[i][c1][c2] = max(dp[i][c1][c2], candidate)
                    
                    # Option 3: op1 + op2 (if c1>0, c2>0 and nums[i-1]>=k)
                    if c1 > 0 and c2 > 0 and saving_12[i-1] > 0:
                        candidate = dp[i-1][c1-1][c2-1] + val_12
                        dp[i][c1][c2] = max(dp[i][c1][c2], candidate)

        total_sum = sum(nums)
        max_saving = dp[n][op1][op2]
        return total_sum - max_saving