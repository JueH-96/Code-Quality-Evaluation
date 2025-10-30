class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Remove duplicates and sort
        rewards = sorted(set(rewardValues))
        
        # Set to track all possible totals we can achieve
        possible = {0}
        
        for reward in rewards:
            # For each current possible total, if it's less than reward,
            # we can add reward to get a new possible total
            new_possible = set()
            for total in possible:
                if total < reward:
                    new_possible.add(total + reward)
            
            # Add all new possible totals to our set
            possible.update(new_possible)
        
        return max(possible)