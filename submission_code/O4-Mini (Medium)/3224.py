from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 10**9 + 7

        # Precompute factorials and inverse factorials up to n
        fac = [1] * (n + 1)
        for i in range(1, n + 1):
            fac[i] = fac[i - 1] * i % MOD

        inv_fac = [1] * (n + 1)
        inv_fac[n] = pow(fac[n], MOD - 2, MOD)
        for i in range(n, 0, -1):
            inv_fac[i - 1] = inv_fac[i] * i % MOD

        # Collect the lengths of the segments of uninfected children
        segments = []
        # Left boundary segment (before the first sick child)
        first_sick = sick[0]
        if first_sick > 0:
            segments.append(first_sick)
        # Interior segments (between sick children)
        for i in range(len(sick) - 1):
            gap = sick[i + 1] - sick[i] - 1
            segments.append(gap)
        # Right boundary segment (after the last sick child)
        last_sick = sick[-1]
        right_len = (n - 1) - last_sick
        if right_len > 0:
            segments.append(right_len)

        # Total number of uninfected children
        total_uninfected = sum(segments)
        if total_uninfected == 0:
            return 1

        # Start with multinomial coefficient: total_uninfected! / (prod segments[i]!)
        result = fac[total_uninfected]
        for seg in segments:
            result = (result * inv_fac[seg]) % MOD

        # For each interior segment (i.e., segments that lie between two initial sick children),
        # multiply by 2^(length-1) to account for the two-sided infection propagation.
        # The interior segments are exactly those gaps computed in the loop above,
        # excluding the first and last in `segments` if they correspond to boundaries.
        # We can re-derive them or else just iterate the original gaps between sick children:
        for i in range(len(sick) - 1):
            gap = sick[i + 1] - sick[i] - 1
            if gap > 0:
                # within this gap, at each of the first (gap-1) steps we have two choices
                result = result * pow(2, gap - 1, MOD) % MOD

        return result