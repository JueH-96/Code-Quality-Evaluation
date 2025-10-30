class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        max_score = nums[0]
        
        # Create a prefix sum array
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Iterate through all possible starting positions
        for i in range(1, n + 1):
            # Calculate the score for the current starting position
            score = prefix_sum[n] - prefix_sum[i]
            
            # Check if the current starting position is valid
            valid = True
            for j in range(i, n):
                if (nums[i - 1] ^ nums[j]) & 1 != 0:
                    score -= x
                    if score < 0:
                        valid = False
                        break
            
            # Update the maximum score
            if valid:
                max_score = max(max_score, score)
        
        return max_score