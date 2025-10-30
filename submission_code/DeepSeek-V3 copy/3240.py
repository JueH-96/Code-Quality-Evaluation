class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def count_prices(num):
            total = 0
            for i in range(1, 64):  # Since 2^63 is a very large number
                if i % x == 0:
                    # Calculate the number of set bits at position i in numbers from 1 to num
                    # The pattern is: every 2^i numbers, the i-th bit is set in 2^(i-1) numbers
                    # So for the i-th bit, the count is (num + 1) // (2^i) * 2^(i-1) + max(0, (num + 1) % (2^i) - 2^(i-1))
                    full_cycles = (num + 1) // (1 << i)
                    remainder = (num + 1) % (1 << i)
                    total += full_cycles * (1 << (i - 1))
                    total += max(0, remainder - (1 << (i - 1)))
            return total

        low = 1
        high = 1 << 60  # A very large number to start with
        result = 0

        while low <= high:
            mid = (low + high) // 2
            current = count_prices(mid)
            if current <= k:
                result = mid
                low = mid + 1
            else:
                high = mid - 1

        return result