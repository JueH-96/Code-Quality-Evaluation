from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Compute previous less for minimum (strictly less)
        prev_less = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] > nums[i]:
                stack.pop()
            prev_less[i] = stack[-1] if stack else -1
            stack.append(i)
        
        # Compute next less or equal for minimum
        next_less_equal = [n] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            next_less_equal[i] = stack[-1] if stack else n
            stack.append(i)
        
        # Compute previous greater for maximum (strictly greater)
        prev_greater = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            prev_greater[i] = stack[-1] if stack else -1
            stack.append(i)
            
        # Compute next greater or equal for maximum
        next_greater_equal = [n] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            next_greater_equal[i] = stack[-1] if stack else n
            stack.append(i)
        
        # Helper function:
        # Given m = min(L, k) and R,
        # compute S = sum_{x=1}^{m} min(R, k - x + 1)
        # Change variable: let t = k - x + 1, so t goes from k-m+1 to k.
        def sum_min_R(m: int, R: int, k: int) -> int:
            if m <= 0:
                return 0
            A = k - m + 1  # lowest t
            B = k         # highest t
            # We need S = sum_{t=A}^{B} min(R, t).
            if B < R:
                # All t are less than R so min(R, t) = t.
                count = B - A + 1
                return (A + B) * count // 2
            elif A >= R:
                # All t are at least R so min(R, t) = R.
                count = B - A + 1
                return R * count
            else:
                # Some t are below R and some not.
                # t from A to R-1: use t; then for t from R to B: use R.
                part1_count = R - A
                s1 = (A + (R - 1)) * part1_count // 2
                part2_count = B - R + 1
                s2 = R * part2_count
                return s1 + s2
        
        total = 0
        
        # For each index i, count how many subarrays (with length <= k)
        # have nums[i] as the minimum and as the maximum.
        for i in range(n):
            # For minimum:
            Lmin = i - prev_less[i]          # number of left choices
            Rmin = next_less_equal[i] - i      # number of right choices
            left_count_min = Lmin if Lmin < k else k
            contrib_min = nums[i] * sum_min_R(left_count_min, Rmin, k)
            
            # For maximum:
            Lmax = i - prev_greater[i]         # number of left choices
            Rmax = next_greater_equal[i] - i     # number of right choices
            left_count_max = Lmax if Lmax < k else k
            contrib_max = nums[i] * sum_min_R(left_count_max, Rmax, k)
            
            total += contrib_min + contrib_max
        
        return total

# For testing when the module is run as a script.
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.minMaxSubarraySum([1,2,3], 2))  # Expected output: 20
    # Example 2
    print(sol.minMaxSubarraySum([1,-3,1], 2))  # Expected output: -6