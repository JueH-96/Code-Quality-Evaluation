class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = {}
        def solve(i, parity):
            if i == n:
                return 0
            if (i, parity) in dp:
                return dp[(i, parity)]
            
            best_score = -float('inf')
            
            #Option 1: Don't move
            best_score = max(best_score, nums[i] + solve(i+1, nums[i]%2))

            #Option 2: Move to next position
            for j in range(i + 1, n):
                score = nums[i]
                if nums[i]%2 != nums[j]%2:
                    score -= x
                best_score = max(best_score, score + solve(j+1, nums[j]%2))

            dp[(i, parity)] = best_score
            return best_score

        return solve(0, nums[0]%2)