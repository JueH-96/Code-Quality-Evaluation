class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        
        # Create points with their sums
        points = []
        for i in range(n):
            points.append((nums1[i], nums2[i], nums1[i] + nums2[i]))
        
        # Sort points by nums1 in descending order
        points.sort(reverse=True)
        
        # Create queries with original indices  
        queries_with_idx = []
        for i in range(len(queries)):
            queries_with_idx.append((queries[i][0], queries[i][1], i))
        
        # Sort queries by x in descending order
        queries_with_idx.sort(reverse=True)
        
        result = [0] * len(queries)
        
        # Maintain dominating points: list of (nums2, sum)
        # No point dominates another in this list
        dominating = []
        point_idx = 0
        
        for x, y, query_idx in queries_with_idx:
            # Add all points with nums1 >= x
            while point_idx < n and points[point_idx][0] >= x:
                nums1_val, nums2_val, sum_val = points[point_idx]
                
                # Check if new point should be added to dominating set
                should_add = True
                new_dominating = []
                
                for d_nums2, d_sum in dominating:
                    if d_nums2 >= nums2_val and d_sum >= sum_val:
                        # New point is dominated, don't add it
                        should_add = False
                        new_dominating.append((d_nums2, d_sum))
                    elif nums2_val >= d_nums2 and sum_val >= d_sum:
                        # New point dominates existing point, remove existing
                        pass
                    else:
                        # Neither dominates, keep existing
                        new_dominating.append((d_nums2, d_sum))
                
                if should_add:
                    new_dominating.append((nums2_val, sum_val))
                
                dominating = new_dominating
                point_idx += 1
            
            # Find maximum sum among dominating points with nums2 >= y
            max_sum = -1
            for nums2_val, sum_val in dominating:
                if nums2_val >= y:
                    max_sum = max(max_sum, sum_val)
            
            result[query_idx] = max_sum
        
        return result