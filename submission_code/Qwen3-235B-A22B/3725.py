from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute left_min and right_min for sum_min_contrib
        left_min = [-1] * n
        stack = [-1]
        for i in range(n):
            while stack[-1] != -1 and nums[stack[-1]] >= nums[i]:
                stack.pop()
            left_min[i] = stack[-1]
            stack.append(i)
        
        right_min = [n] * n
        stack = [n]
        for i in reversed(range(n)):
            while stack[-1] != n and nums[stack[-1]] > nums[i]:
                stack.pop()
            right_min[i] = stack[-1]
            stack.append(i)
        
        # Compute left_max and right_max for sum_max_contrib
        left_max = [-1] * n
        stack = [-1]
        for i in range(n):
            while stack[-1] != -1 and nums[stack[-1]] <= nums[i]:
                stack.pop()
            left_max[i] = stack[-1]
            stack.append(i)
        
        right_max = [n] * n
        stack = [n]
        for i in reversed(range(n)):
            while stack[-1] != n and nums[stack[-1]] < nums[i]:
                stack.pop()
            right_max[i] = stack[-1]
            stack.append(i)
        
        # Compute sum_min_contrib
        sum_min = 0
        for i in range(n):
            low = left_min[i] + 1
            high = right_min[i] - 1
            a = max(low, i - k + 1)
            b = i
            if a > high:
                contrib = 0
            else:
                high_val = right_min[i] - 1
                split_x = high_val - k + 1
                x1 = a
                x2 = min(b, split_x)
                sum_case1 = 0
                if x1 <= x2:
                    terms = x2 - x1 + 1
                    total_x = (x1 + x2) * terms // 2
                    sum_case1 = total_x + terms * (k - i)
                
                x3 = max(a, split_x + 1)
                sum_case2 = 0
                if x3 <= b:
                    terms = b - x3 + 1
                    y_count = high_val - i + 1
                    if y_count < 0:
                        y_count = 0
                    sum_case2 = terms * y_count
                
                count = sum_case1 + sum_case2
            sum_min += nums[i] * count
        
        # Compute sum_max_contrib
        sum_max = 0
        for i in range(n):
            low = left_max[i] + 1
            high = right_max[i] - 1
            a = max(low, i - k + 1)
            b = i
            if a > high:
                contrib = 0
            else:
                high_val = right_max[i] - 1
                split_x = high_val - k + 1
                x1 = a
                x2 = min(b, split_x)
                sum_case1 = 0
                if x1 <= x2:
                    terms = x2 - x1 + 1
                    total_x = (x1 + x2) * terms // 2
                    sum_case1 = total_x + terms * (k - i)
                
                x3 = max(a, split_x + 1)
                sum_case2 = 0
                if x3 <= b:
                    terms = b - x3 + 1
                    y_count = high_val - i + 1
                    if y_count < 0:
                        y_count = 0
                    sum_case2 = terms * y_count
                
                count = sum_case1 + sum_case2
            sum_max += nums[i] * count
        
        return sum_min + sum_max