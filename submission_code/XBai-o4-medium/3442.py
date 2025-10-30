import bisect
from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort the reward values
        rewardValues.sort()
        # Deduplicate the reward values while maintaining order
        unique_rewards = []
        prev = -1
        for r in rewardValues:
            if r != prev:
                unique_rewards.append(r)
                prev = r
        
        # Initialize the existing sums with 0
        existing_sums = [0]
        
        for s in unique_rewards:
            # Find the insertion point for s in existing_sums
            m = bisect.bisect_left(existing_sums, s)
            # Generate B = [x + s for x in existing_sums[:m]
            B = [x + s for x in existing_sums[:m]]
            # existing_part is the part of existing_sums from m onwards
            part_a = existing_sums[m:]
            part_b = B
            # Merge part_a and part_b into a sorted list with unique elements
            merged = []
            i = j = 0
            while i < len(part_a) and j < len(part_b):
                if part_a[i] < part_b[j]:
                    merged.append(part_a[i])
                    i += 1
                elif part_a[i] > part_b[j]:
                    merged.append(part_b[j])
                    j += 1
                else:
                    merged.append(part_a[i])
                    i += 1
                    j += 1
            # Add remaining elements from part_a and part_b
            while i < len(part_a):
                merged.append(part_a[i])
                i += 1
            while j < len(part_b):
                merged.append(part_b[j])
                j += 1
            # Update existing_sums
            existing_sums = existing_sums[:m] + merged
        
        return existing_sums[-1]