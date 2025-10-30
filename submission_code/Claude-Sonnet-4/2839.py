class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        
        # Sort queries by x in descending order, keep track of original indices
        sorted_queries = sorted(enumerate(queries), key=lambda x: x[1][0], reverse=True)
        
        # Sort points by nums1 in descending order
        points = sorted(range(n), key=lambda i: nums1[i], reverse=True)
        
        result = [0] * len(queries)
        stack = []  # Stack of (nums2[i], nums1[i] + nums2[i]) in decreasing order of nums2
        j = 0
        
        for query_idx, (x, y) in sorted_queries:
            # Add all points with nums1[i] >= x to our stack
            while j < n and nums1[points[j]] >= x:
                idx = points[j]
                curr_nums2 = nums2[idx]
                curr_sum = nums1[idx] + nums2[idx]
                
                # Remove points from stack that are dominated by current point
                while stack and stack[-1][1] <= curr_sum:
                    stack.pop()
                
                # Add current point if it's not dominated
                if not stack or stack[-1][0] > curr_nums2:
                    stack.append((curr_nums2, curr_sum))
                
                j += 1
            
            # Find maximum sum among points with nums2[i] >= y
            max_sum = -1
            for nums2_val, sum_val in stack:
                if nums2_val >= y:
                    max_sum = max(max_sum, sum_val)
            
            result[query_idx] = max_sum
        
        return result