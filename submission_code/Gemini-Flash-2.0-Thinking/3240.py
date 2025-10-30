class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def sum_of_prices_efficient(num, x):
            total_price = 0
            max_bits = num.bit_length() + 2
            for j in range(1, max_bits):
                if j % x == 0:
                    power_of_2 = 1 << (j - 1)
                    cycle_len = 1 << j
                    num_cycles = num // cycle_len
                    remainder = num % cycle_len

                    count = num_cycles * power_of_2
                    if remainder >= power_of_2:
                        count += remainder - power_of_2 + 1
                    total_price += count
            return total_price

        low = 1
        high = 2 * k + 100  # A generous upper bound
        ans = 0

        while low <= high:
            mid = (low + high) // 2
            current_sum = sum_of_prices_efficient(mid, x)

            if current_sum <= k:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1

        return ans