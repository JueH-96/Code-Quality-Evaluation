from bisect import bisect_left

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        
        # Create points (nums1, nums2, sum)
        points = [(nums1[i], nums2[i], nums1[i] + nums2[i]) for i in range(n)]
        # Sort by nums1 in descending order
        points.sort(reverse=True)
        
        # Add index to queries and sort by x in descending order
        indexed_queries = [(queries[i][0], queries[i][1], i) for i in range(len(queries))]
        indexed_queries.sort(reverse=True)
        
        answer = [-1] * len(queries)
        skyline = []  # List of (nums2, max_sum) pairs
        
        point_idx = 0
        for x, y, query_idx in indexed_queries:
            # Add all points with nums1 >= x to the skyline
            while point_idx < n and points[point_idx][0] >= x:
                _, nums2_val, sum_val = points[point_idx]
                point_idx += 1
                
                # Find position to insert based on nums2 value
                pos = bisect_left(skyline, (nums2_val, float('inf')))
                
                # Check if the new point is dominated
                if pos > 0 and skyline[pos - 1][1] >= sum_val:
                    continue
                
                # Remove points dominated by the new point
                # Remove points to the right with sum <= sum_val
                while pos < len(skyline) and skyline[pos][1] <= sum_val:
                    skyline.pop(pos)
                
                # Remove points to the left with sum <= sum_val (to maintain decreasing sum)
                while pos > 0 and skyline[pos - 1][1] <= sum_val:
                    skyline.pop(pos - 1)
                    pos -= 1
                
                # Insert the new point
                skyline.insert(pos, (nums2_val, sum_val))
            
            # Query for the maximum sum with nums2 >= y
            pos = bisect_left(skyline, (y, -1))
            if pos < len(skyline):
                answer[query_idx] = skyline[pos][1]
        
        return answer