class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff_indices = [i for i in range(n) if s1[i] != s2[i]]
        num_diff = len(diff_indices)

        if num_diff % 2 != 0:
            return -1

        if num_diff == 0:
            return 0

        dp = {}

        def solve(indices):
            if not indices:
                return 0

            if tuple(sorted(indices)) in dp:
                return dp[tuple(sorted(indices))]

            if len(indices) == 2:
                return x

            min_cost = float('inf')
            first = indices[0]
            remaining = indices[1:]

            for i, second in enumerate(remaining):
                other_indices = list(remaining)
                other_indices.pop(i)
                cost = x + solve(other_indices)
                min_cost = min(min_cost, cost)

            # Consider adjacent flips
            if len(indices) >= 2:
                min_cost_adj = float('inf')
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1 = indices[i]
                        idx2 = indices[j]

                        cost_direct = x

                        # Try to resolve by adjacent flips
                        if abs(idx1 - idx2) == 1:
                            cost_adj = 1
                            remaining_indices = list(indices)
                            remaining_indices.pop(max(i,j))
                            remaining_indices.pop(min(i,j))
                            cost_adj += solve(remaining_indices)
                            min_cost_adj = min(min_cost_adj, cost_adj)

                min_cost = min(min_cost, min_cost_adj if min_cost_adj != float('inf') else float('inf'))

            dp[tuple(sorted(indices))] = min_cost
            return min_cost

        # Let's try DP with the number of differences
        memo = {}
        diff_indices.sort()

        def dp_solve(diffs):
            if not diffs:
                return 0
            if tuple(diffs) in memo:
                return memo[tuple(diffs)]

            if len(diffs) == 2:
                return x

            min_cost = float('inf')
            first = diffs[0]
            for i in range(1, len(diffs)):
                second = diffs[i]
                remaining_diffs = list(diffs[1:])
                remaining_diffs.pop(i - 1)
                cost = x + dp_solve(tuple(remaining_diffs))
                min_cost = min(min_cost, cost)

            memo[tuple(diffs)] = min_cost
            return min_cost

        n_diff = len(diff_indices)
        dp_perfect_match = {}

        def find_min_cost(current_diffs):
            if not current_diffs:
                return 0

            if tuple(current_diffs) in dp_perfect_match:
                return dp_perfect_match[tuple(current_diffs)]

            min_cost = float('inf')
            idx1 = current_diffs[0]
            remaining_diffs = list(current_diffs[1:])

            for idx2 in remaining_diffs:
                cost = x + find_min_cost(tuple(d for d in remaining_diffs if d != idx2))
                min_cost = min(min_cost, cost)

            dp_perfect_match[tuple(current_diffs)] = min_cost
            return min_cost

        if num_diff > 0:
            dp2 = [float('inf')] * (1 << num_diff)
            dp2[0] = 0

            for mask in range(1 << num_diff):
                if dp2[mask] == float('inf'):
                    continue

                current_indices = []
                for i in range(num_diff):
                    if (mask >> i) & 1:
                        current_indices.append(diff_indices[i])

                if len(current_indices) % 2 != 0:
                    continue

                if len(current_indices) >= 2:
                    for i in range(len(current_indices)):
                        for j in range(i + 1, len(current_indices)):
                            idx1 = current_indices[i]
                            idx2 = current_indices[j]

                            next_mask = mask ^ (1 << diff_indices.index(idx1)) ^ (1 << diff_indices.index(idx2))
                            dp2[next_mask] = min(dp2[next_mask], dp2[mask] + x)

                            if abs(idx1 - idx2) == 1:
                                next_mask_adj = mask ^ (1 << diff_indices.index(idx1)) ^ (1 << diff_indices.index(idx2))
                                dp2[next_mask_adj] = min(dp2[next_mask_adj], dp2[mask] + 1)

            return dp2[0] if num_diff == 0 else dp2[(1 << num_diff) - 1]
        else:
            return 0