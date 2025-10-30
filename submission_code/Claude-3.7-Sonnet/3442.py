class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        n = len(rewardValues)
        
        # Memoization table
        memo = {}
        
        def solve(mask, total):
            # If we've already computed this state, return the result
            if (mask, total) in memo:
                return memo[(mask, total)]
            
            # Initialize max_reward with current total
            max_reward = total
            
            # Try to choose each unmarked index
            for i in range(n):
                # Check if index i is unmarked and its value is greater than current total
                if (mask & (1 << i)) == 0 and rewardValues[i] > total:
                    # Mark the index and add its value to total
                    new_mask = mask | (1 << i)
                    new_total = total + rewardValues[i]
                    
                    # Recursively find the maximum reward
                    max_reward = max(max_reward, solve(new_mask, new_total))
            
            # Store result in memoization table
            memo[(mask, total)] = max_reward
            return max_reward
        
        return solve(0, 0)