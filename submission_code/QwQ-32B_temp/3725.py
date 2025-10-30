from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        L_min = self.compute_L_min(nums)
        R_min = self.compute_R_min(nums)
        L_max = self.compute_L_max(nums)
        R_max = self.compute_R_max(nums)
        
        total_min = 0
        total_max = 0
        
        for i in range(n):
            count_min = self.compute_count_min(i, L_min, R_min, k, n)
            total_min += nums[i] * count_min
            count_max = self.compute_count_max(i, L_max, R_max, k, n)
            total_max += nums[i] * count_max
        
        return total_min + total_max
    
    def compute_L_min(self, nums: List[int]) -> List[int]:
        n = len(nums)
        L = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            if stack:
                L[i] = stack[-1]
            else:
                L[i] = -1
            stack.append(i)
        return L
    
    def compute_R_min(self, nums: List[int]) -> List[int]:
        n = len(nums)
        R = [n] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if stack:
                R[i] = stack[-1]
            else:
                R[i] = n
            stack.append(i)
        return R
    
    def compute_L_max(self, nums: List[int]) -> List[int]:
        n = len(nums)
        L = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                L[i] = stack[-1]
            else:
                L[i] = -1
            stack.append(i)
        return L
    
    def compute_R_max(self, nums: List[int]) -> List[int]:
        n = len(nums)
        R = [n] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                R[i] = stack[-1]
            else:
                R[i] = n
            stack.append(i)
        return R
    
    def compute_count_min(self, i: int, L: List[int], R: List[int], k: int, n: int) -> int:
        L_val = L[i]
        R_val = R[i]
        s_min = L_val + 1
        s_max = i
        part1 = 0
        part2 = 0
        
        s_end_part1 = min(s_max, R_val - k - 1)
        s_start_part1 = s_min
        if s_start_part1 <= s_end_part1:
            num_terms = s_end_part1 - s_start_part1 + 1
            sum_s = (s_start_part1 + s_end_part1) * num_terms // 2
            term = (k - i) * num_terms
            part1 = sum_s + term
        
        s_start_part2 = max(s_min, R_val - k)
        s_end_part2 = s_max
        if s_start_part2 <= s_end_part2:
            num_terms = s_end_part2 - s_start_part2 + 1
            part2 = (R_val - i) * num_terms
        
        return part1 + part2
    
    def compute_count_max(self, i: int, L: List[int], R: List[int], k: int, n: int) -> int:
        L_val = L[i]
        R_val = R[i]
        s_min = L_val + 1
        s_max = i
        part1 = 0
        part2 = 0
        
        s_end_part1 = min(s_max, R_val - k - 1)
        s_start_part1 = s_min
        if s_start_part1 <= s_end_part1:
            num_terms = s_end_part1 - s_start_part1 + 1
            sum_s = (s_start_part1 + s_end_part1) * num_terms // 2
            term = (k - i) * num_terms
            part1 = sum_s + term
        
        s_start_part2 = max(s_min, R_val - k)
        s_end_part2 = s_max
        if s_start_part2 <= s_end_part2:
            num_terms = s_end_part2 - s_start_part2 + 1
            part2 = (R_val - i) * num_terms
        
        return part1 + part2