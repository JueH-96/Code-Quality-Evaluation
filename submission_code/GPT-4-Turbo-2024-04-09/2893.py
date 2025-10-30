class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # dp[i] will store the maximum score achievable starting from index i
        dp = [0] * n
        dp[-1] = nums[-1]  # Base case: the last element's score is just its own value
        
        # Fill dp array from right to left
        for i in range(n-2, -1, -1):
            max_score = nums[i]
            current_score = nums[i]
            for j in range(i+1, n):
                # Calculate the penalty if parities are different
                penalty = x if (nums[i] % 2) != (nums[j] % 2) else 0
                # Update the current score considering the jump to position j
                current_score = nums[i] + dp[j] - penalty
                # Update the maximum score from position i
                max_score = max(max_score, current_score)
            dp[i] = max_score
        
        return dp[0]