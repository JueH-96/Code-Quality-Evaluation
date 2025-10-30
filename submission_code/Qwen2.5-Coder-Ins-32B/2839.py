from typing import List
from collections import defaultdict
import bisect

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Combine nums1 and nums2 into a list of pairs and sort by nums1 in descending order
        pairs = sorted(zip(nums1, nums2), reverse=True)
        
        # Sort queries by x in descending order and keep track of original indices
        sorted_queries = sorted(enumerate(queries), key=lambda x: -x[1][0])
        
        # Dictionary to store the max sums for each y value
        y_to_max_sum = defaultdict(list)
        
        # Result array initialized with -1
        result = [-1] * len(queries)
        
        # Index to track position in pairs
        pair_index = 0
        
        for query_index, (x, y) in sorted_queries:
            # Process pairs that have nums1 >= x
            while pair_index < len(pairs) and pairs[pair_index][0] >= x:
                num1, num2 = pairs[pair_index]
                current_sum = num1 + num2
                # Insert current_sum into the list for y_to_max_sum[num2] while maintaining sorted order
                bisect.insort(y_to_max_sum[num2], current_sum)
                pair_index += 1
            
            # Find the maximum sum for the current query
            max_sum = -1
            for key in y_to_max_sum:
                if key >= y:
                    # Binary search to find the largest sum >= y
                    index = bisect.bisect_left(y_to_max_sum[key], y)
                    if index < len(y_to_max_sum[key]):
                        max_sum = max(max_sum, y_to_max_sum[key][index])
            
            result[query_index] = max_sum
        
        return result