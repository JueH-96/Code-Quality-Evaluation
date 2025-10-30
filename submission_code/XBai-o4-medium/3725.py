from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        S = k - 1
        
        # Compute for max
        pg = self.get_prev_greater(nums)
        ng = self.get_next_greater_or_equal(nums)
        
        # Compute for min
        ps = self.get_prev_smaller(nums)
        ns = self.get_next_smaller_or_equal(nums)
        
        sum_max = 0
        sum_min = 0
        
        for i in range(n):
            # For max
            L_max = i - pg[i]
            R_max = ng[i] - i
            cnt_max = self.count_pairs(L_max, R_max, S)
            sum_max += nums[i] * cnt_max
            
            # For min
            L_min = i - ps[i]
            R_min = ns[i] - i
            cnt_min = self.count_pairs(L_min, R_min, S)
            sum_min += nums[i] * cnt_min
            
        return sum_max + sum_min
        
    def get_prev_greater(self, nums):
        n = len(nums)
        pg = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                pg[i] = stack[-1]
            else:
                pg[i] = -1
            stack.append(i)
        return pg
    
    def get_next_greater_or_equal(self, nums):
        n = len(nums)
        ng = [n] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            if stack:
                ng[i] = stack[-1]
            else:
                ng[i] = n
            stack.append(i)
        return ng
    
    def get_prev_smaller(self, nums):
        n = len(nums)
        ps = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            if stack:
                ps[i] = stack[-1]
            else:
                ps[i] = -1
            stack.append(i)
        return ps
    
    def get_next_smaller_or_equal(self, nums):
        n = len(nums)
        ns = [n] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if stack:
                ns[i] = stack[-1]
            else:
                ns[i] = n
            stack.append(i)
        return ns
    
    def count_pairs(self, A, B, S):
        if S >= A + B - 1:
            return A * B
        if S < 0:
            return 0
        part1_a_max = S - (B - 1)
        part1_a_max = min(part1_a_max, A-1)
        if part1_a_max < 0:
            part1 = 0
        else:
            part1 = (part1_a_max + 1) * B

        a_min_part2 = max(0, S - (B - 1) + 1)
        a_max_part2 = min(A-1, S)
        if a_min_part2 > a_max_part2:
            part2 = 0
        else:
            n_terms = a_max_part2 - a_min_part2 + 1
            sum_a = (a_min_part2 + a_max_part2) * n_terms // 2
            part2 = (S + 1)*n_terms - sum_a
        return part1 + part2