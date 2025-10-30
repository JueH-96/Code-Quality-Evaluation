class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Create pairs of nums1, nums2 and sort by nums1 in descending order
        pairs = sorted([(n1, n2) for n1, n2 in zip(nums1, nums2)], reverse=True)
        
        # Create queries with index and sort by x in descending order
        queries_with_index = sorted([(x, y, i) for i, (x, y) in enumerate(queries)], reverse=True)
        
        stack = []  # monotonic stack storing (nums2, sum) pairs
        answer = [-1] * len(queries)
        j = 0
        
        for x, y, i in queries_with_index:
            # Process all pairs where nums1 >= x
            while j < len(pairs) and pairs[j][0] >= x:
                n1, n2 = pairs[j]
                curr_sum = n1 + n2
                
                # Maintain monotonic decreasing stack of nums2 values
                # If we find a larger sum for a smaller nums2, we can remove the previous entry
                while stack and stack[-1][1] <= curr_sum:
                    if stack[-1][0] <= n2:
                        stack.pop()
                    else:
                        break
                        
                if not stack or stack[-1][0] > n2:
                    stack.append((n2, curr_sum))
                j += 1
            
            # Binary search to find the first element >= y
            left, right = 0, len(stack)
            while left < right:
                mid = (left + right) // 2
                if stack[mid][0] >= y:
                    right = mid
                else:
                    left = mid + 1
                    
            if left < len(stack):
                answer[i] = stack[left][1]
                
        return answer