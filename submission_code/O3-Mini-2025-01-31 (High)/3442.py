from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # We use a dynamic programming approach with a bitset.
        # The idea is to let dp be an integer where the i­th bit is set 
        # if there exists a valid chain of operations resulting in a total reward equal to i.
        # A valid chain is one where we choose rewards in (non‐decreasing) order
        # and each chosen reward must be greater than the current total.
        # (Note: when arranged in increasing order, any valid chain must be strictly increasing.)
        #
        # Initially, our total reward x is 0, so only bit 0 is set.
        dp = 1  # binary: ...0001 means a sum of 0 is possible.
        
        # Sort the rewards.
        # We do not explicitly remove duplicates since if a reward occurs more than once,
        # it still cannot be added twice in the chain (because the chain must be strictly increasing).
        for reward in sorted(rewardValues):
            # When we are considering a candidate reward, we can only add it to a chain
            # whose current total s is strictly less than reward.
            # In our bitset dp, the bits corresponding to s (0 <= s < reward) that are set can be "extended"
            # by using the reward. We compute a mask to extract these possibilities.
            allowed = dp & ((1 << reward) - 1)
            # For each such valid sum s (s < reward), including reward leads to a new sum s + reward.
            # We update our dp with these new sums by shifting allowed left by reward and OR–ing it in.
            dp |= allowed << reward

        # Finally, the maximum total reward achievable is the highest sum s for which dp has a set bit.
        # We can get that by using the bit_length() method on dp.
        return dp.bit_length() - 1