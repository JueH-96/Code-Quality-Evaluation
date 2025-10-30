class Solution:
    def minMaxSubarraysSum(self, nums, k):
        n = len(nums)
        if n == 0:
            return 0
        
        left_min = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            left_min[i] = i - stack[-1] if stack else i + 1
            stack.append(i)
        
        right_min = [0] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            right_min[i] = stack[-1] - i if stack else n - i
            stack.append(i)
        
        left_max = [0] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            left_max[i] = i - stack[-1] if stack else i + 1
            stack.append(i)
        
        right_max = [0] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            right_max[i] = stack[-1] - i if stack else n - i
            stack.append(i)
        
        T = k - 1
        total_min_sum = 0
        total_max_sum = 0
        
        for i in range(n):
            A_min = left_min[i] - 1
            B_min = right_min[i] - 1
            total_pairs_min = (A_min + 1) * (B_min + 1)
            if T >= A_min + B_min:
                violating_min = 0
            else:
                a_low = max(0, T + 1 - B_min)
                a_high1 = min(A_min, T)
                seg1 = 0
                if a_low <= a_high1:
                    n1 = a_high1 - a_low + 1
                    sum_a = (a_low + a_high1) * n1 // 2
                    seg1 = n1 * (B_min - T) + sum_a
                a_low2 = max(a_low, T + 1)
                seg2 = 0
                if a_low2 <= A_min:
                    n2 = A_min - a_low2 + 1
                    seg2 = n2 * (B_min + 1)
                violating_min = seg1 + seg2
            count_min_i = total_pairs_min - violating_min
            total_min_sum += nums[i] * count_min_i
            
            A_max = left_max[i] - 1
            B_max = right_max[i] - 1
            total_pairs_max = (A_max + 1) * (B_max + 1)
            if T >= A_max + B_max:
                violating_max = 0
            else:
                a_low = max(0, T + 1 - B_max)
                a_high1 = min(A_max, T)
                seg1 = 0
                if a_low <= a_high1:
                    n1 = a_high1 - a_low + 1
                    sum_a = (a_low + a_high1) * n1 // 2
                    seg1 = n1 * (B_max - T) + sum_a
                a_low2 = max(a_low, T + 1)
                seg2 = 0
                if a_low2 <= A_max:
                    n2 = A_max - a_low2 + 1
                    seg2 = n2 * (B_max + 1)
                violating_max = seg1 + seg2
            count_max_i = total_pairs_max - violating_max
            total_max_sum += nums[i] * count_max_i
        
        return total_min_sum + total_max_sum