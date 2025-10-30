from bisect import bisect_left
from typing import List

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        # Build dictionary mapping each value to all its indices
        index_map = {}
        for idx, val in enumerate(nums):
            if val not in index_map:
                index_map[val] = []
            index_map[val].append(idx)
        
        results = []
        # For each query, process the result
        for query_index in queries:
            val = nums[query_index]
            indices = index_map[val]
            # if only one occurrence, append -1
            if len(indices) < 2:
                results.append(-1)
                continue
            
            # Find the position of query_index in indices (it is sorted)
            pos = bisect_left(indices, query_index)
            
            # There are two neighbors in the circular sorted list.
            # previous neighbor: indices[ (pos-1) % len(indices) ]
            # next neighbor: indices[ (pos+1) % len(indices) ] (ensuring mod behavior)
            prev_index = indices[pos - 1]  # works even if pos=0, gives last element
            next_index = indices[(pos + 1) % len(indices)]  # if at last element, gets first element
            
            # calculate distances in a circular manner
            # distance from query_index to neighbor index j is:
            # diff = abs(query_index - j) and circular_distance = min(diff, n - diff)
            # compute for both previous and next neighbors
            def circular_distance(i, j):
                diff = abs(i - j)
                return min(diff, n - diff)
            
            dist_prev = circular_distance(query_index, prev_index)
            dist_next = circular_distance(query_index, next_index)
            
            results.append(min(dist_prev, dist_next))
        
        return results
            
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    nums = [1,3,1,4,1,3,2]
    queries = [0,3,5]
    print(sol.solveQueries(nums, queries))  # Expected output: [2, -1, 3]
    
    # Example 2:
    nums = [1,2,3,4]
    queries = [0,1,2,3]
    print(sol.solveQueries(nums, queries))  # Expected output: [-1, -1, -1, -1]