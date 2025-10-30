class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Compute left_ORs: OR values for selecting k elements from the first i+1 elements
        left_ORs = []
        prev_dp = [set() for _ in range(k + 1)]
        prev_dp[0].add(0)
        for num in nums:
            new_dp = [set() for _ in range(k + 1)]
            for c in range(k + 1):
                new_dp[c] = set(prev_dp[c])  # Copy previous state without current num
            for c in range(1, k + 1):
                for or_val in prev_dp[c - 1]:
                    new_dp[c].add(or_val | num)
            prev_dp = new_dp
            left_ORs.append(prev_dp[k].copy())
        
        # Compute right_ORs: OR values for selecting k elements from elements starting at index i to the end
        right_ORs = [set() for _ in range(n)]
        prev_dp = [set() for _ in range(k + 1)]
        prev_dp[0].add(0)
        for i in range(n - 1, -1, -1):
            num = nums[i]
            new_dp = [set() for _ in range(k + 1)]
            for c in range(k + 1):
                new_dp[c] = set(prev_dp[c])  # Copy previous state without current num
            for c in range(1, k + 1):
                for or_val in prev_dp[c - 1]:
                    new_dp[c].add(or_val | num)
            prev_dp = new_dp
            right_ORs[i] = prev_dp[k].copy()
        
        max_val = 0
        # Iterate over all valid split points m
        for m in range(k - 1, n - k):
            left = left_ORs[m]
            right = right_ORs[m + 1]
            if not left or not right:
                continue
            current_max = max(a ^ b for a in left for b in right)
            if current_max > max_val:
                max_val = current_max
        
        return max_val