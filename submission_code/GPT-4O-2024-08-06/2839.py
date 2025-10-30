from typing import List
from bisect import bisect_left

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        # Combine nums1 and nums2 into a list of tuples with their sum
        combined = [(nums1[i], nums2[i], nums1[i] + nums2[i]) for i in range(n)]
        # Sort combined by nums1[i] descending, and if equal by nums2[i] descending
        combined.sort(reverse=True, key=lambda x: (x[0], x[1]))
        
        # Prepare to answer queries
        results = []
        # Sort queries by x_i descending, and if equal by y_i descending
        queries_with_index = sorted([(queries[i][0], queries[i][1], i) for i in range(len(queries))], reverse=True)
        
        # Use a list to keep track of the best sums for each nums2[j] value
        best_sums = []
        
        # Iterate over the queries
        j = 0
        for x_i, y_i, original_index in queries_with_index:
            # Add all elements from combined that have nums1[j] >= x_i
            while j < n and combined[j][0] >= x_i:
                # Insert the current nums2[j] and its sum into best_sums
                nums2_j, sum_j = combined[j][1], combined[j][2]
                # Insert in sorted order by nums2[j]
                pos = bisect_left(best_sums, (nums2_j, float('-inf')))
                if pos < len(best_sums) and best_sums[pos][0] == nums2_j:
                    # If the same nums2[j] exists, keep the maximum sum
                    best_sums[pos] = (nums2_j, max(best_sums[pos][1], sum_j))
                else:
                    best_sums.insert(pos, (nums2_j, sum_j))
                j += 1
            
            # Find the maximum sum for nums2[j] >= y_i
            pos = bisect_left(best_sums, (y_i, float('-inf')))
            if pos < len(best_sums):
                results.append((original_index, best_sums[pos][1]))
            else:
                results.append((original_index, -1))
        
        # Sort results back to the original query order
        results.sort()
        return [result[1] for result in results]