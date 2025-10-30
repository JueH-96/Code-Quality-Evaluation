from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Precompute left boundaries: for each index, find previous index with value > nums[i]
        left = [0] * n
        stack = []
        for i in range(n):
            # Pop until we find element > nums[i] (not >= because equal does not break maximum condition)
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            # If stack empty, no previous greater exists
            left[i] = stack[-1] + 1 if stack else 0
            stack.append(i)
            
        # Precompute right boundaries: for each index, next index with value > nums[i]
        right = [0] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] < nums[i]:
                # strictly less; note: for equal values, we do not break the valid interval,
                # because if two equal v's are present, they do not “protect” against each other.
                stack.pop()
            right[i] = stack[-1] - 1 if stack else n - 1
            stack.append(i)
            
        # Group all indices by their value.
        from collections import defaultdict
        positions_dict = defaultdict(list)
        for i, val in enumerate(nums):
            positions_dict[val].append(i)
        
        ans = 0
        # For each distinct value v,
        # only subarrays where the maximum is v count.
        # But note: a subarray that lies fully in the intersection for some occurrence of v
        # automatically has no element > v. So we count over v.
        for v, pos_list in positions_dict.items():
            if len(pos_list) < k:
                continue
            # pos_list is sorted
            # We slide a window of size k on pos_list.
            # For each block, compute L_possible = max(left[p] for p in block)
            # and R_possible = min(right[p] for p in block).
            # Because all these indices come from positions with the same value v,
            # you might expect their left and right boundaries to be similar – but they can differ.
            # We compute these explicitly.
            # A simple method: for each window, iterate over the k indices. k is at most 10^5,
            # and the sum over all windows is O(n) overall.
            # (If performance is an issue one could precompute RMQ, but here constraints allow it.)
            m = len(pos_list)
            # To ease iterative computation, we can use a sliding window technique.
            # But note that k can be large, but the total number of occurrences for v is less.
            for i in range(m - k + 1):
                # For the window pos_list[i] ... pos_list[i+k-1]:
                L_possible = left[pos_list[i]]
                R_possible = right[pos_list[i]]
                # update over the k block:
                for j in range(i, i+k):
                    if left[pos_list[j]] > L_possible:
                        L_possible = left[pos_list[j]]
                    if right[pos_list[j]] < R_possible:
                        R_possible = right[pos_list[j]]
                # It must be that L_possible <= pos_list[i] and R_possible >= pos_list[i+k-1]
                if L_possible <= pos_list[i] and R_possible >= pos_list[i+k-1]:
                    cnt_left = pos_list[i] - L_possible + 1
                    cnt_right = R_possible - pos_list[i+k-1] + 1
                    ans += cnt_left * cnt_right
        return ans

# Below is a simple test harness.
if __name__ == '__main__':
    sol = Solution()
    # Example 1
    nums1 = [1,3,2,3,3]
    k1 = 2
    print(sol.countSubarrays(nums1, k1))  # Expected output: 6

    # Example 2
    nums2 = [1,4,2,1]
    k2 = 3
    print(sol.countSubarrays(nums2, k2))  # Expected output: 0