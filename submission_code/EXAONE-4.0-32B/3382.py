from collections import defaultdict

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        left_greater = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                left_greater[i] = stack[-1]
            else:
                left_greater[i] = -1
            stack.append(i)
        
        right_greater = [n] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                right_greater[i] = stack[-1]
            else:
                right_greater[i] = n
            stack.append(i)
        
        count_dict = defaultdict(int)
        for i in range(n):
            x = nums[i]
            l_bound = left_greater[i] + 1
            r_bound = right_greater[i] - 1
            count_dict[(x, l_bound, r_bound)] += 1
        
        ans = 0
        for count in count_dict.values():
            ans += count * (count + 1) // 2
        
        return ans