class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        max_len = 1
        dp = []
        max_so_far = []
        for j in range(n):
            dp_j = [0] * 301
            for k in range(j):
                d = abs(nums[j] - nums[k])
                if k >= len(max_so_far):
                    continue
                max_val_k = max_so_far[k][d]
                candidate = max_val_k + 1
                if candidate == 1:
                    candidate = 2
                if candidate > dp_j[d]:
                    dp_j[d] = candidate
                if dp_j[d] > max_len:
                    max_len = dp_j[d]
            # Compute max_so_far_j for j
            max_so_far_j = [0] * 301
            current_max = 0
            for d in range(300, -1, -1):
                current_max = max(current_max, dp_j[d])
                max_so_far_j[d] = current_max
            dp.append(dp_j)
            max_so_far.append(max_so_far_j)
        return max_len if max_len >= 1 else 1