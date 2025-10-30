class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        max_value = 300
        dp = [[0] * 300 for _ in range(max_value + 1)]  # dp[value][difference]
        suffix_max = [[0] * 300 for _ in range(max_value + 1)]
        present = [False] * (max_value + 1)
        overall_max = 1

        for x in nums:
            temp_dp = [0] * 300
            current_max_candidate = 0

            for y in range(1, max_value + 1):
                if present[y]:
                    d_new = abs(x - y)
                    if d_new >= 300:
                        continue  # since max difference is 299, this is redundant but safe

                    max_prev_with_d_ge = suffix_max[y][d_new]
                    candidate_length = max(max_prev_with_d_ge, 1) + 1

                    if candidate_length > temp_dp[d_new]:
                        temp_dp[d_new] = candidate_length
                        if candidate_length > current_max_candidate:
                            current_max_candidate = candidate_length

            # Update dp[x]
            for d in range(300):
                if temp_dp[d] > dp[x][d]:
                    dp[x][d] = temp_dp[d]

            # Compute suffix_max for x
            suffix_max_x = [0] * 300
            max_so_far = 0
            for d in range(299, -1, -1):
                max_so_far = max(max_so_far, dp[x][d])
                suffix_max_x[d] = max_so_far
            suffix_max[x] = suffix_max_x

            # Update overall_max
            current_max = max(current_max_candidate, 1)
            if current_max > overall_max:
                overall_max = current_max

            # Mark x as present
            present[x] = True

        return overall_max