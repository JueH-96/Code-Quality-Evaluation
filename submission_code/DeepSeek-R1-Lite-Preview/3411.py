class Solution:
    def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
        from math import log2, floor

        def total_set_bits(n):
            count = 0
            k = 0
            while (1 << k) <= n:
                full_cycles = n // (1 << (k + 1))
                count += full_cycles * (1 << k)
                remainder = n % (1 << (k + 1))
                count += max(0, remainder - (1 << k) + 1)
                k += 1
            return count

        def find_N(pos):
            left = 1
            right = pos
            while left < right:
                mid = (left + right) // 2
                if total_set_bits(mid) < pos:
                    left = mid + 1
                else:
                    right = mid
            return left

        def get_exponent(pos):
            N = find_N(pos)
            set_bits_up_to_N_minus_1 = total_set_bits(N - 1)
            index_in_N = pos - set_bits_up_to_N_minus_1 - 1
            set_bits = []
            for k in range(60):  # since 2^60 is larger than 10^15
                if (N >> k) & 1:
                    set_bits.append(k)
            return set_bits[index_in_N]

        def sum_exponents(pos):
            if pos == 0:
                return 0
            sum_exp = 0
            left = 1
            right = pos
            while left <= right:
                mid = (left + right) // 2
                N_mid = find_N(mid)
                set_bits_up_to_N_mid_minus_1 = total_set_bits(N_mid - 1)
                index_in_N_mid = mid - set_bits_up_to_N_mid_minus_1 - 1
                set_bits_N_mid = []
                for k in range(60):
                    if (N_mid >> k) & 1:
                        set_bits_N_mid.append(k)
                exponent = set_bits_N_mid[index_in_N_mid]
                sum_exp += exponent
                right = mid - 1
            return sum_exp

        answer = []
        for q in queries:
            from_i, to_i, mod_i = q
            if from_i > to_i:
                product_mod = 1
            else:
                sum_to = sum_exponents(to_i)
                sum_from = sum_exponents(from_i - 1)
                total_sum = sum_to - sum_from
                product_mod = pow(2, total_sum, mod_i)
            answer.append(product_mod)
        return answer