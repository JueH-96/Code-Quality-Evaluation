class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        q = len(queries)
        
        # Create pairs of (nums1[j], nums2[j], sum)
        pairs = [(nums1[j], nums2[j], nums1[j] + nums2[j]) for j in range(n)]
        
        # Sort pairs by nums1 in descending order
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Sort queries by x_i in descending order, keeping track of original indices
        indexed_queries = [(i, queries[i][0], queries[i][1]) for i in range(q)]
        indexed_queries.sort(key=lambda x: x[1], reverse=True)
        
        result = [-1] * q
        stack = []  # Monotonic stack to track maximum sums
        pair_idx = 0
        
        for query_idx, x_i, y_i in indexed_queries:
            # Process all pairs where nums1[j] >= x_i
            while pair_idx < n and pairs[pair_idx][0] >= x_i:
                _, nums2_val, sum_val = pairs[pair_idx]
                
                # Maintain a monotonic stack where as nums2 increases, sum increases
                # Remove entries that are dominated by the current pair
                while stack and stack[-1][0] <= nums2_val and stack[-1][1] <= sum_val:
                    stack.pop()
                
                # Add current pair if it's not dominated
                if not stack or stack[-1][1] < sum_val:
                    stack.append((nums2_val, sum_val))
                
                pair_idx += 1
            
            # Find maximum sum for pairs where nums2[j] >= y_i
            max_sum = -1
            for nums2_val, sum_val in stack:
                if nums2_val >= y_i:
                    max_sum = sum_val
                    break
            
            result[query_idx] = max_sum
        
        return result