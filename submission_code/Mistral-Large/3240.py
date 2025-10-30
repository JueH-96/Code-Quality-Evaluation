class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_set_bits_at_positions_divisible_by_x(num, x):
            binary_representation = bin(num)[2:]
            count = 0
            for i in range(len(binary_representation)):
                if (i + 1) % x == 0 and binary_representation[i] == '1':
                    count += 1
            return count

        total_price = 0
        num = 0

        while total_price <= k:
            num += 1
            total_price += count_set_bits_at_positions_divisible_by_x(num, x)
            if total_price > k:
                return num - 1

        return num