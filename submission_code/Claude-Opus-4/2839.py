class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Create list of tuples (nums1[i], nums2[i], sum)
        n = len(nums1)
        points = []
        for i in range(n):
            points.append((nums1[i], nums2[i], nums1[i] + nums2[i]))
        
        # Sort points by nums1 in descending order
        points.sort(key=lambda x: -x[0])
        
        # Process queries with their original indices
        indexed_queries = [(queries[i][0], queries[i][1], i) for i in range(len(queries))]
        # Sort queries by x in descending order
        indexed_queries.sort(key=lambda x: -x[0])
        
        result = [-1] * len(queries)
        
        # Use a monotonic stack to maintain maximum sums
        # Stack will store (y, max_sum) pairs
        stack = []
        point_idx = 0
        
        for x, y, query_idx in indexed_queries:
            # Add all points with nums1[i] >= x to our consideration
            while point_idx < n and points[point_idx][0] >= x:
                curr_x, curr_y, curr_sum = points[point_idx]
                
                # Maintain monotonic property: if current y is greater than stack top's y,
                # we only keep it if its sum is also greater
                while stack and stack[-1][0] <= curr_y and stack[-1][1] <= curr_sum:
                    stack.pop()
                
                # Only add if it provides a better sum for its y value
                if not stack or stack[-1][1] < curr_sum:
                    stack.append((curr_y, curr_sum))
                
                point_idx += 1
            
            # Binary search in stack to find the best sum where y >= query's y
            left, right = 0, len(stack) - 1
            max_sum = -1
            
            while left <= right:
                mid = (left + right) // 2
                if stack[mid][0] >= y:
                    max_sum = stack[mid][1]
                    right = mid - 1
                else:
                    left = mid + 1
            
            result[query_idx] = max_sum
        
        return result