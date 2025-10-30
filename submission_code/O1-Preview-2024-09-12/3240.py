class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def compute_total_price(n, x):
            total_price = 0
            maxbit = n.bit_length()
            for k in range(maxbit):
                if (k + 1) % x == 0:
                    cycles = (n + 1) // (1 << (k + 1))
                    ones_in_full_cycles = cycles * (1 << k)
                    remainder = (n + 1) % (1 << (k + 1))
                    ones_in_remainder = max(0, remainder - (1 << k))
                    ones_in_remainder = min(ones_in_remainder, (1 << k))
                    total_ones_at_k = ones_in_full_cycles + ones_in_remainder
                    total_price += total_ones_at_k
            return total_price

        left, right = 1, 10**17
        answer = 0
        while left <= right:
            mid = (left + right) // 2
            total_price = compute_total_price(mid, x)
            if total_price <= k:
                answer = mid
                left = mid + 1
            else:
                right = mid - 1
        return answer