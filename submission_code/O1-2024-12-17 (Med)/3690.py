class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        import math

        n = len(s)
        # Gather the lengths of consecutive runs of '0's or '1's
        runs = []
        current_run_length = 1
        for i in range(1, n):
            if s[i] == s[i-1]:
                current_run_length += 1
            else:
                runs.append(current_run_length)
                current_run_length = 1
        runs.append(current_run_length)

        # If the entire string has all runs of length <= 1 already:
        max_run = max(runs)

        # Edge case: if max_run == 1, we cannot do better than 1
        # (the longest substring of identical chars is already 1).
        if max_run == 1:
            return 1

        # Function that tells us how many flips are needed so that
        # a run of length L has no sub-run longer than X.
        #
        # Idea: Flipping K bits "removes" those K bits from the run,
        # leaving L-K bits unflipped. We can group those L-K unflipped bits
        # into (K+1) segments. Each segment can be at most X in length. So
        # we need (K+1)*X >= (L-K). The minimal K satisfying this is our cost.
        def flips_needed(L, X):
            # If L is already <= X, no flips needed
            if L <= X:
                return 0
            # Solve (K+1)*X >= (L - K)  =>  K >= (L - X) / (X+1)
            return math.ceil((L - X) / (X + 1))

        # Check feasibility: can we ensure all runs have length <= X
        # using at most numOps flips in total?
        def can_achieve(X):
            total_flips = 0
            for length in runs:
                total_flips += flips_needed(length, X)
                if total_flips > numOps:
                    return False
            return total_flips <= numOps

        # Binary search for the smallest X in [1..max_run] such that can_achieve(X) is True
        left, right = 1, max_run
        answer = max_run

        while left <= right:
            mid = (left + right) // 2
            if can_achieve(mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1

        return answer