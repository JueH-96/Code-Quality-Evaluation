class Solution:
    def minMaxSubarraySum(self, nums, k):
        n = len(nums)
        if n == 0:
            return 0
        
        left_max = [-1] * n
        right_max = [n] * n
        left_min = [-1] * n
        right_min = [n] * n
        
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                left_max[i] = stack[-1]
            else:
                left_max[i] = -1
            stack.append(i)
        
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            if stack:
                right_max[i] = stack[-1]
            else:
                right_max[i] = n
            stack.append(i)
        
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            if stack:
                left_min[i] = stack[-1]
            else:
                left_min[i] = -1
            stack.append(i)
        
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if stack:
                right_min[i] = stack[-1]
            else:
                right_min[i] = n
            stack.append(i)
        
        total_max = 0
        total_min = 0
        
        for i in range(n):
            p = left_max[i]
            q = right_max[i]
            L1 = max(p + 1, i - k + 1)
            R1 = min(i, q - k)
            L2 = max(p + 1, q - k + 1)
            R2 = i
            
            count_max_i = 0
            if L1 <= R1:
                num1 = R1 - L1 + 1
                term1 = num1 * (k - i)
                term2 = (L1 + R1) * num1 // 2
                count_max_i += term1 + term2
            if L2 <= R2:
                num2 = R2 - L2 + 1
                count_max_i += num2 * (q - i)
            
            total_max += nums[i] * count_max_i
            
            p_min = left_min[i]
            q_min = right_min[i]
            L1_min = max(p_min + 1, i - k + 1)
            R1_min = min(i, q_min - k)
            L2_min = max(p_min + 1, q_min - k + 1)
            R2_min = i
            
            count_min_i = 0
            if L1_min <= R1_min:
                num1_min = R1_min - L1_min + 1
                term1_min = num1_min * (k - i)
                term2_min = (L1_min + R1_min) * num1_min // 2
                count_min_i += term1_min + term2_min
            if L2_min <= R2_min:
                num2_min = R2_min - L2_min + 1
                count_min_i += num2_min * (q_min - i)
            
            total_min += nums[i] * count_min_i
        
        return total_max + total_min