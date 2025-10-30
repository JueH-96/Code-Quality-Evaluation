from collections import defaultdict
import bisect

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        value_indices = defaultdict(list)
        for idx, num in enumerate(nums):
            value_indices[num].append(idx)
        
        n = len(nums)
        answer = []
        
        for query in queries:
            target_value = nums[query]
            indices = value_indices.get(target_value, [])
            if len(indices) < 2:
                answer.append(-1)
                continue
            
            # Use binary search to find the insertion point
            pos = bisect.bisect_left(indices, query)
            candidates = []
            if pos > 0:
                candidates.append(indices[pos - 1])
            if pos < len(indices):
                # Handle case where query is exactly at indices[pos]
                if indices[pos] == query:
                    if pos + 1 < len(indices):
                        candidates.append(indices[pos + 1])
                else:
                    candidates.append(indices[pos])
            
            min_dist = float('inf')
            for idx in candidates:
                if idx != query:
                    dist = abs(idx - query)
                    circular_dist = min(dist, n - dist)
                    if circular_dist < min_dist:
                        min_dist = circular_dist
            # Also need to check the circular neighbors (wrap around)
            # For example, the first and last elements in indices could be close in circular manner
            # So check the first and last elements in the indices list
            first_idx = indices[0]
            last_idx = indices[-1]
            for idx in [first_idx, last_idx]:
                if idx != query:
                    dist = abs(idx - query)
                    circular_dist = min(dist, n - dist)
                    if circular_dist < min_dist:
                        min_dist = circular_dist
            answer.append(min_dist if min_dist != float('inf') else -1)
        
        return answer