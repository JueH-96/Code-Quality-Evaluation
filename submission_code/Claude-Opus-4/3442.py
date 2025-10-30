class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort rewards to process smaller values first
        rewardValues.sort()
        
        # Use a set to track all possible reward totals
        # Start with 0 as the initial total
        possible_totals = {0}
        
        for reward in rewardValues:
            # For each reward, find all totals we can achieve by adding it
            new_totals = set()
            for total in possible_totals:
                # We can only add this reward if it's greater than current total
                if reward > total:
                    new_totals.add(total + reward)
            
            # Add new totals to our set of possible totals
            possible_totals.update(new_totals)
        
        # Return the maximum possible total
        return max(possible_totals)