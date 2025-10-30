class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        import bisect

        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]

        # dp[i] is the maximum number of splits up to index i
        dp = [0] * (n + 1)
        # last_sum[i] is the sum of the last segment in the optimal split up to i
        last_sum = [0] * (n + 1)

        candidates = []
        # Initialize with (sum_j_plus_last, j)
        candidates.append((0, 0))  # j=0, sum_j_plus_last = 0 + 0

        max_k = 0

        for i in range(1, n + 1):
            # Find the largest j where sum_j_plus_last <= prefix[i]
            idx = bisect.bisect_right(candidates, (prefix[i], float('inf'))) - 1

            if idx >= 0:
                j = candidates[idx][1]
                dp[i] = dp[j] + 1
                last_sum[i] = prefix[i] - prefix[j]
                sum_i_plus_last = prefix[i] + last_sum[i]
                # Insert into candidates
                pos = bisect.bisect_left(candidates, (sum_i_plus_last, i))
                candidates.insert(pos, (sum_i_plus_last, i))
                max_k = max(max_k, dp[i])
            else:
                # No valid split, take the entire subarray up to i as one segment
                dp[i] = 1
                last_sum[i] = prefix[i]
                sum_i_plus_last = prefix[i] + last_sum[i]
                pos = bisect.bisect_left(candidates, (sum_i_plus_last, i))
                candidates.insert(pos, (sum_i_plus_last, i))
                max_k = max(max_k, dp[i])

        return max_k