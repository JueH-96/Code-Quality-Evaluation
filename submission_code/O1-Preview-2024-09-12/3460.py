class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10 ** 9 + 7
        MAX_INV = 0
        req_dict = {}
        for end_i, cnt_i in requirements:
            req_dict[end_i] = cnt_i
            MAX_INV = max(MAX_INV, cnt_i)
        MAX_INV += n * (n - 1) // 2  # upper bound on inversion counts

        dp = [0] * (MAX_INV + 1)
        dp[0] = 1

        for pos in range(1, n + 1):
            max_inv = pos * (pos - 1) // 2
            new_dp = [0] * (MAX_INV + 1)
            prefix_sum = [0] * (MAX_INV + 2)  # extra one for easier indexing

            # Build prefix sums
            for inv in range(MAX_INV + 1):
                prefix_sum[inv + 1] = (prefix_sum[inv] + dp[inv]) % MOD

            # Compute new dp
            for inv in range(MAX_INV + 1):
                min_k = max(0, inv - (pos - 1))
                max_k = inv
                if min_k > max_k:
                    continue
                total = (prefix_sum[inv + 1] - prefix_sum[min_k]) % MOD
                new_dp[inv] = total

            # Apply requirement if any
            if pos - 1 in req_dict:
                cnt_i = req_dict[pos - 1]
                tmp_dp = [0] * (MAX_INV + 1)
                tmp_dp[cnt_i] = new_dp[cnt_i]
                new_dp = tmp_dp

            dp = new_dp

        # Get the inversion count for the whole permutation
        total_perm_inv = None
        for end_i, cnt_i in requirements:
            if end_i == n - 1:
                total_perm_inv = cnt_i
                break

        if total_perm_inv is None:
            return 0

        return dp[total_perm_inv] % MOD