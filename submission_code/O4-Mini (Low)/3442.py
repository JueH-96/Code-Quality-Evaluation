from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort rewards to consider smaller ones first
        rewardValues.sort()
        # Use a sorted list of reachable sums, starting with 0
        reachable = [0]
        seen = set([0])
        max_reward = 0
        
        # For each reward v, try to extend any reachable sum s < v
        for v in rewardValues:
            new_sums = []
            # reachable is sorted ascending
            for s in reachable:
                if s < v:
                    s2 = s + v
                    # if we haven't seen s2 before, schedule to add
                    if s2 not in seen:
                        seen.add(s2)
                        new_sums.append(s2)
                        if s2 > max_reward:
                            max_reward = s2
                else:
                    # since reachable is sorted, no later sums < v
                    break
            # merge new_sums into our reachable list, keeping it sorted
            if new_sums:
                # simple merge since both lists sorted
                merged = []
                i = j = 0
                new_sums.sort()
                while i < len(reachable) and j < len(new_sums):
                    if reachable[i] < new_sums[j]:
                        merged.append(reachable[i])
                        i += 1
                    else:
                        merged.append(new_sums[j])
                        j += 1
                # append leftovers
                if i < len(reachable):
                    merged.extend(reachable[i:])
                if j < len(new_sums):
                    merged.extend(new_sums[j:])
                reachable = merged
        
        return max_reward