class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        dp = [{} for _ in range(k + 1)]
        for num in nums:
            new_dp = [d.copy() for d in dp]
            for c in range(k + 1):
                for x in dp[c]:
                    new_c = c + (0 if x == num else 1)
                    if new_c > k:
                        continue
                    if new_dp[new_c].get(num, 0) < dp[c][x] + 1:
                        new_dp[new_c][num] = dp[c][x] + 1
            # Handle the case where the subsequence starts with this num
            new_dp[0][num] = max(new_dp[0].get(num, 0), 1)
            dp = new_dp
        max_len = 0
        for c in range(k + 1):
            if dp[c]:
                current_max = max(dp[c].values())
                if current_max > max_len:
                    max_len = current_max
        return max_len