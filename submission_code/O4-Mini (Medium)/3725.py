from typing import List

class Solution:
    def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        K = k - 1
        
        # Helper to compute counts of subarrays for each position i
        # such that nums[i] is the minimum (or maximum) and length ≤ k.
        def count_pairs(A: int, B: int) -> int:
            # count of integer pairs (a,b) with 0 ≤ a < A, 0 ≤ b < B, a + b ≤ K
            if K < 0:
                return 0
            # if the constraint a+b ≤ K is loose
            if K >= A + B - 2:
                return A * B
            # otherwise sum over a in [0..min(A-1,K)]
            t = min(A - 1, K)
            # split at where K - a ≥ B-1
            c = K - (B - 1)
            # for a ≤ c, all b in [0..B-1] are allowed
            if c >= 0:
                m = min(c, t)
                count1 = (m + 1) * B
            else:
                count1 = 0
            # for a in [max(0,c+1)..t], b ≤ K - a
            low = max(0, c + 1)
            high = t
            if low > high:
                count2 = 0
            else:
                n2 = high - low + 1
                # sum_{a=low..high} (K - a + 1)
                # = n2 * ((K+1 - low) + (K+1 - high)) / 2
                count2 = n2 * ((K + 1 - low) + (K + 1 - high)) // 2
            return count1 + count2
        
        # Compute prev_less and next_less_equal for minima
        prev_less = [-1] * n
        next_less_equal = [n] * n
        stack = []
        # prev_less: previous index with nums[j] < nums[i]
        for i in range(n):
            while stack and nums[stack[-1]] >= nums[i]:
                stack.pop()
            prev_less[i] = stack[-1] if stack else -1
            stack.append(i)
        # next_less_equal: next index with nums[j] <= nums[i]
        stack.clear()
        for i in range(n):
            while stack and nums[i] <= nums[stack[-1]]:
                j = stack.pop()
                next_less_equal[j] = i
            stack.append(i)
        
        # Compute prev_greater and next_greater_equal for maxima
        prev_greater = [-1] * n
        next_greater_equal = [n] * n
        stack.clear()
        # prev_greater: previous index with nums[j] > nums[i]
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            prev_greater[i] = stack[-1] if stack else -1
            stack.append(i)
        # next_greater_equal: next index with nums[j] >= nums[i]
        stack.clear()
        for i in range(n):
            while stack and nums[i] >= nums[stack[-1]]:
                j = stack.pop()
                next_greater_equal[j] = i
            stack.append(i)
        
        total = 0
        # Sum contributions of each nums[i] as minimum
        for i in range(n):
            A = i - prev_less[i]
            B = next_less_equal[i] - i
            cnt = count_pairs(A, B)
            total += nums[i] * cnt
        
        # Sum contributions of each nums[i] as maximum
        for i in range(n):
            A = i - prev_greater[i]
            B = next_greater_equal[i] - i
            cnt = count_pairs(A, B)
            total += nums[i] * cnt
        
        return total