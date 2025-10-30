from typing import List
from collections import defaultdict
from itertools import combinations

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])
        
        # Create a dictionary to store the maximum weight for each end time
        max_weight = defaultdict(int)
        
        # Iterate through the intervals
        for start, end, weight in intervals:
            # Find the maximum weight up to the current start time
            max_weight_so_far = max(max_weight.get(i, 0) for i in range(start))
            
            # Update the maximum weight for the current end time
            max_weight[end] = max(max_weight[end], max_weight_so_far + weight)
        
        # Find the 4 intervals with the maximum total weight
        max_weight_list = sorted(max_weight.items(), key=lambda x: x[1], reverse=True)
        result = []
        for i, (_, weight) in enumerate(max_weight_list):
            if len(result) < 4:
                result.append(i)
            else:
                break
        
        # Find the lexicographically smallest array of indices
        intervals_sorted = sorted(enumerate(intervals), key=lambda x: (x[1][0], x[1][1], -x[1][2]))
        return [intervals_sorted[i][0] for i in result]