class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        from typing import List

        import sys
        sys.setrecursionlimit(1 << 25)

        def cum_counts_sums(x):
            total_ones = 0
            total_positions = 0
            max_bit = x.bit_length()
            for b in range(max_bit):
                cycle_len = 1 << (b + 1)
                complete_cycles = x // cycle_len
                ones_from_complete_cycles = complete_cycles * (1 << b)
                remainder = x % cycle_len
                ones_from_remainder = max(0, remainder - (1 << b) + 1)
                total_ones_at_b = ones_from_complete_cycles + ones_from_remainder
                total_ones += total_ones_at_b
                total_positions += total_ones_at_b * b
            return total_ones, total_positions

        def positions_of_set_bits(x):
            positions = []
            i = 0
            while x > 0:
                if x & 1:
                    positions.append(i)
                x >>=1
                i +=1
            return positions

        def cum_exponents(pos):
            if pos < 0:
                return 0
            # Binary search x such that cum_counts(x) > pos
            left = 1
            right = 1 << 55  # Since pos can be up to 1e15
            while left < right:
                mid = (left + right) // 2
                cum_counts_mid, _ = cum_counts_sums(mid)
                if cum_counts_mid > pos:
                    right = mid
                else:
                    left = mid + 1
            x = left
            cum_counts_x_minus1, cum_sums_x_minus1 = cum_counts_sums(x - 1)
            idx_in_powerful_x = pos - cum_counts_x_minus1
            positions = []
            x_temp = x
            i = 0
            while x_temp > 0:
                if x_temp &1:
                    positions.append(i)
                x_temp >>=1
                i +=1
            positions.sort()
            sum_of_positions = sum(positions[:idx_in_powerful_x + 1])
            cum_exponents_pos = cum_sums_x_minus1 + sum_of_positions
            return cum_exponents_pos

        answer = []
        for from_i, to_i, mod_i in queries:
            expo_from = cum_exponents(from_i - 1)
            expo_to = cum_exponents(to_i)
            total_exponents = expo_to - expo_from
            result = pow(2, total_exponents, mod_i)
            answer.append(result)
        return answer