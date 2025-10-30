from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort the rewards so that we consider them in increasing order.
        # Note: The condition for selecting a reward r is that it must be greater than the current total,
        # and because the chosen rewards can be reordered arbitrarily (the condition forces an increasing order),
        # an optimal solution is equivalent to choosing a subset that can be arranged in ascending order such that,
        # if we denote the chosen rewards as r1, r2, …, rk then r1 > 0, r2 > r1, r3 > (r1+r2), etc.
        #
        # We can use dynamic programming to keep track of all achievable total rewards (x values)
        # by processing each reward exactly once. For each reward r (taken from the sorted list),
        # for every achievable total s so far, if r > s then we can take this reward (if not already taken)
        # and get a new total s + r.
        #
        # dp will be a set of achievable totals.
        
        rewardValues.sort()
        dp = {0}  # starting total is 0
        
        # Process each reward once.
        for r in rewardValues:
            # We iterate over the snapshot of current achievable totals.
            new_totals = set()
            for s in dp:
                if r > s:
                    new_totals.add(s + r)
            # Merge new_totals into dp.
            dp |= new_totals
        
        # The answer is the maximum total reward achievable.
        return max(dp)
        
# Example usage:
# sol = Solution()
# print(sol.maxTotalReward([1,1,3,3]))  # Expected output: 4
# print(sol.maxTotalReward([1,6,4,3,2]))  # Expected output: 11