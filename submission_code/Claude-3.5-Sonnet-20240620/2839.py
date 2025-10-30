class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Sort pairs (nums1[i], nums2[i], nums1[i] + nums2[i]) by nums1[i] in descending order
        sorted_nums = sorted([(n1, n2, n1 + n2) for n1, n2 in zip(nums1, nums2)], reverse=True)
        
        # Sort queries with their indices
        sorted_queries = sorted([(x, y, i) for i, (x, y) in enumerate(queries)], reverse=True)
        
        stack = []
        answer = [-1] * len(queries)
        j = 0
        
        for x, y, i in sorted_queries:
            # Process all numbers >= x
            while j < len(sorted_nums) and sorted_nums[j][0] >= x:
                n1, n2, sum_val = sorted_nums[j]
                
                # Maintain a monotonic decreasing stack of (nums2, sum)
                while stack and stack[-1][1] <= sum_val:
                    stack.pop()
                
                if not stack or stack[-1][0] < n2:
                    stack.append((n2, sum_val))
                
                j += 1
            
            # Binary search for the first element >= y
            idx = bisect.bisect_left(stack, (y, float('inf')))
            if idx < len(stack):
                answer[i] = stack[idx][1]
        
        return answer