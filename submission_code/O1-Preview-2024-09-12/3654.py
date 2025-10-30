class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        n = len(nums)
        initial_sum = sum(nums)
        max_op1 = op1
        max_op2 = op2

        dp = [[-float('inf')] * (max_op2 + 1) for _ in range(max_op1 + 1)]
        dp[0][0] = 0

        for idx in range(n):
            num = nums[idx]
            R_op1 = num // 2  # Reduction from Op1 only
            R_op2 = k if num >= k else 0  # Reduction from Op2 only
            if num >= k:
                R_both = k + (num - k) // 2  # Reduction from both operations
            else:
                R_both = 0

            dp_new = [row[:] for row in dp]
            for op1_used in range(max_op1 +1):
                for op2_used in range(max_op2 +1):
                    if dp[op1_used][op2_used] == -float('inf'):
                        continue
                    curr_reduction = dp[op1_used][op2_used]

                    # Option 0: Do nothing
                    dp_new[op1_used][op2_used] = max(dp_new[op1_used][op2_used], curr_reduction)

                    # Option 1: Apply Op1 only
                    if op1_used +1 <= max_op1:
                        dp_new[op1_used +1][op2_used] = max(dp_new[op1_used +1][op2_used],
                                                             curr_reduction + R_op1)

                    # Option 2: Apply Op2 only
                    if op2_used +1 <= max_op2:
                        dp_new[op1_used][op2_used +1] = max(dp_new[op1_used][op2_used +1],
                                                             curr_reduction + R_op2)

                    # Option 3: Apply both
                    if num >= k and op1_used +1 <= max_op1 and op2_used +1 <= max_op2:
                        dp_new[op1_used +1][op2_used +1] = max(dp_new[op1_used +1][op2_used +1],
                                                                curr_reduction + R_both)
            dp = dp_new

        max_reduction = 0
        for op1_used in range(max_op1 +1):
            for op2_used in range(max_op2 +1):
                max_reduction = max(max_reduction, dp[op1_used][op2_used])

        min_possible_sum = initial_sum - max_reduction
        return min_possible_sum