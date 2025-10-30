from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        # Precompute binary lifting tables up to the highest bit in k
        max_bit = k.bit_length()  # we will use bits [0..max_bit-1]

        # nexts[j][i] = node you reach from i by 2^j steps
        # sums[j][i] = sum of the node IDs along that 2^j-step path (excluding the starting node i)
        nexts = [receiver[:] ]  # nexts[0]
        sums = [receiver[:] ]   # sums[0]

        # Build tables for j = 1 .. max_bit-1
        for j in range(1, max_bit):
            prev_next = nexts[j-1]
            prev_sum = sums[j-1]
            cur_next = [0] * n
            cur_sum = [0] * n
            for i in range(n):
                # first jump 2^(j-1) from i goes to mid
                mid = prev_next[i]
                # then another 2^(j-1) from mid
                dest = prev_next[mid]
                cur_next[i] = dest
                # sum of the two half-paths
                cur_sum[i] = prev_sum[i] + prev_sum[mid]
            nexts.append(cur_next)
            sums.append(cur_sum)

        # Pre-extract which bits of k are 1
        bits = [b for b in range(max_bit) if (k >> b) & 1]

        ans = 0
        # For each possible starting player x
        for x in range(n):
            total = x
            cur = x
            # jump according to bits of k
            for b in bits:
                total += sums[b][cur]
                cur = nexts[b][cur]
            if total > ans:
                ans = total

        return ans