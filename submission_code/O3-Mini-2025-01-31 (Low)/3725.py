from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Helper function that, given left and right counts,
        # returns the number of subarrays (using pairs (L,R)) 
        # with L in [1, left], R in [1, right] and L+R <= (k+1).
        def count_restrict(left: int, right: int, k: int) -> int:
            count = 0
            # L from 1 to left (each representing how many elements 
            # from the left (including the current element) are taken)
            for L in range(1, left+1):
                # For given L, R must satisfy:
                #    R <= min(right, k+1 - L)
                maxR = k + 1 - L
                if maxR < 1:
                    break  # no valid R if k+1-L < 1
                count += min(right, maxR)
            return count
        
        # For subarray minima: use monotonic increasing stack.
        left_min = [0]*n
        right_min = [0]*n
        
        stack = []
        # prev less (strict): for each i, index of previous element < nums[i]
        # so left_min[i] = i - prev_index
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            if stack:
                left_min[i] = i - stack[-1]
            else:
                left_min[i] = i + 1
            stack.append(i)
        
        stack = []
        # next less or equal: for each i, next index j with nums[j] <= nums[i]
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            if stack:
                right_min[i] = stack[-1] - i
            else:
                right_min[i] = n - i
            stack.append(i)
        
        # For subarray maxima: similar logic but reverse comparisons.
        left_max = [0]*n
        right_max = [0]*n
        
        stack = []
        # prev greater (strict): for each i, previous index with nums[j] > nums[i]
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            if stack:
                left_max[i] = i - stack[-1]
            else:
                left_max[i] = i + 1
            stack.append(i)
        
        stack = []
        # next greater or equal: for each i, next index j with nums[j] >= nums[i]
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                right_max[i] = stack[-1] - i
            else:
                right_max[i] = n - i
            stack.append(i)
        
        total_min = 0
        total_max = 0
        
        # For each element, count its contribution as min and as max.
        for i in range(n):
            cnt_min = count_restrict(left_min[i], right_min[i], k)
            cnt_max = count_restrict(left_max[i], right_max[i], k)
            total_min += nums[i] * cnt_min
            total_max += nums[i] * cnt_max
        
        return total_min + total_max

# If you want to run a quick test:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minMaxSubarraySum([1,2,3], 2))  # Expected output: 20
    print(sol.minMaxSubarraySum([1,-3,1], 2))  # Expected output: -6