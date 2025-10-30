class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        pairs = sorted(zip(nums1, nums2), reverse=True)
        
        queries_with_index = [(x, y, i) for i, (x, y) in enumerate(queries)]
        queries_with_index.sort(key=lambda q: q[0], reverse=True)
        
        answer = [-1] * len(queries)
        
        stack = []  # (nums2[j], nums1[j] + nums2[j])
        
        j = 0
        for x, y, i in queries_with_index:
            # Add valid pairs to the stack
            while j < n and pairs[j][0] >= x:
                num2 = pairs[j][1]
                sum_val = pairs[j][0] + pairs[j][1]
                
                # Maintain a decreasing stack of sums
                while stack and stack[-1][1] <= sum_val:
                    stack.pop()
                
                stack.append((num2, sum_val))
                j += 1
            
            # Binary search for the best sum
            l, r = 0, len(stack) - 1
            best_sum = -1
            
            while l <= r:
                mid = (l + r) // 2
                if stack[mid][0] >= y:
                    best_sum = stack[mid][1]
                    l = mid + 1
                else:
                    r = mid - 1
            
            answer[i] = best_sum
        
        return answer