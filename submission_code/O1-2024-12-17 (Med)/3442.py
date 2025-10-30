from typing import List
import bisect

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort the reward values so we can process them in ascending order
        rewardValues.sort()
        
        # We maintain a sorted list of all possible sums we can achieve.
        # Initially, we can only achieve the sum = 0 (no rewards picked).
        possible_sums = [0]
        
        # For each reward r in ascending order:
        #  1. Find how many of the current possible sums are strictly less than r.
        #  2. Those sums can be extended by picking this reward (since r > sum_so_far).
        #  3. Merge these new sums (sum_so_far + r) back into the list of possible sums.
        for r in rewardValues:
            # idx is the position up to which the sums are < r
            idx = bisect.bisect_left(possible_sums, r)
            
            # Create new sums by adding r to each sum that is strictly less than r
            new_sums = [s + r for s in possible_sums[:idx]]
            
            # Now we merge new_sums into possible_sums in sorted order, removing duplicates
            merged = []
            i, j = 0, 0
            while i < len(possible_sums) and j < len(new_sums):
                if possible_sums[i] < new_sums[j]:
                    # Add possible_sums[i] if it's not a duplicate of the last appended
                    if not merged or merged[-1] != possible_sums[i]:
                        merged.append(possible_sums[i])
                    i += 1
                elif possible_sums[i] > new_sums[j]:
                    # Add new_sums[j] if it's not a duplicate of the last appended
                    if not merged or merged[-1] != new_sums[j]:
                        merged.append(new_sums[j])
                    j += 1
                else:
                    # They are equal, add one of them (avoiding duplicates)
                    if not merged or merged[-1] != possible_sums[i]:
                        merged.append(possible_sums[i])
                    i += 1
                    j += 1
            
            # Append any remaining values from either list
            while i < len(possible_sums):
                if not merged or merged[-1] != possible_sums[i]:
                    merged.append(possible_sums[i])
                i += 1
            while j < len(new_sums):
                if not merged or merged[-1] != new_sums[j]:
                    merged.append(new_sums[j])
                j += 1
            
            # Update possible_sums with the merged array
            possible_sums = merged
        
        # The maximum achievable sum will be the last (largest) element
        return possible_sums[-1]