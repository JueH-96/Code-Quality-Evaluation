class Solution:
    def minimumMoves(self, nums: list, k: int, maxChanges: int) -> int:
        n = len(nums)
        min_total = float('inf')

        for i in range(n):
            initial = 1 if nums[i] == 1 else 0
            if initial > 0:
                needed = k - 1
            else:
                needed = k

            if needed <= 0:
                if min_total > 0:
                    min_total = 0
                continue

            # Collect existing 1s (excluding the initial one if present)
            existing = []
            for j in range(n):
                if j == i:
                    continue
                if nums[j] == 1:
                    distance = abs(j - i)
                    existing.append(distance)
            existing.sort()

            # Collect zeros, calculate their cost (distance + 1)
            zeros = []
            for j in range(n):
                if nums[j] == 0:
                    cost = abs(j - i) + 1  # flip + swap
                    zeros.append(cost)
            zeros.sort()

            # Precompute prefix sums for existing and zeros
            prefix_existing = [0]
            for d in existing:
                prefix_existing.append(prefix_existing[-1] + d)

            prefix_zeros = [0]
            for c in zeros:
                prefix_zeros.append(prefix_zeros[-1] + c)

            # Try all possible m (number of existing 1s used)
            max_m = min(len(existing), needed)
            for m in range(0, max_m + 1):
                sum_e = prefix_existing[m]
                remaining = needed - m
                if remaining <= 0:
                    total = sum_e
                else:
                    # Determine the maximum t we can take from zeros
                    max_t = min(remaining, len(zeros), maxChanges)
                    if max_t > 0:
                        sum_z = prefix_zeros[max_t]
                        total = sum_e + sum_z
                    else:
                        continue  # not enough to cover remaining
                if total < min_total:
                    min_total = total

            # Also consider cases where we use only flips
            max_t_only = min(needed, len(zeros), maxChanges)
            if max_t_only > 0:
                sum_z_only = prefix_zeros[max_t_only]
                if sum_z_only < min_total:
                    min_total = sum_z_only

        return min_total if min_total != float('inf') else 0