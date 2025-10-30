import sys

class Solution:
    def minSizeSubarray(self, nums, target):
        n = len(nums)
        total = sum(nums)
        
        ans = float('inf')
        l = 0
        current_sum = 0
        for r in range(n):
            current_sum += nums[r]
            while current_sum > target and l <= r:
                current_sum -= nums[l]
                l += 1
            if current_sum == target:
                ans = min(ans, r - l + 1)
        
        if ans == float('inf'):
            ans = 10**18
        
        if total == 0:
            return -1
        
        prefix_arr = [0] * n
        prefix_arr[0] = nums[0]
        for i in range(1, n):
            prefix_arr[i] = prefix_arr[i-1] + nums[i]
        
        prefix_map = {}
        for j in range(n):
            s_val = prefix_arr[j]
            if s_val not in prefix_map:
                prefix_map[s_val] = j
            else:
                if j < prefix_map[s_val]:
                    prefix_map[s_val] = j
        
        if 0 in prefix_map:
            prefix_map[0] = -1
        else:
            prefix_map[0] = -1
        
        suffix_arr = [0] * n
        suffix_arr[-1] = nums[-1]
        for i in range(n-2, -1, -1):
            suffix_arr[i] = suffix_arr[i+1] + nums[i]
        
        M = target // total
        for k in range(0, M+1):
            if k * n + 1 >= ans:
                break
            x = target - k * total
            if x < 0:
                break
            if x == 0:
                candidate = k * n
                if candidate < ans:
                    ans = candidate
            else:
                f_val = 10**18
                for i in range(n):
                    needed = x - suffix_arr[i]
                    if needed in prefix_map:
                        j = prefix_map[needed]
                        length_seg = n - i + j + 1
                        if length_seg < f_val:
                            f_val = length_seg
                if f_val < 10**18:
                    candidate = k * n + f_val
                    if candidate < ans:
                        ans = candidate
        
        return ans if ans != 10**18 else -1