class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def compute_sum(N: int) -> int:
            total = 0
            i = x
            while True:
                bit = 1 << (i - 1)
                if bit > N:
                    break
                cycle = bit * 2
                cnt = (N + 1) // cycle * bit
                remainder = (N + 1) % cycle
                if remainder > bit:
                    cnt += remainder - bit
                total += cnt
                i += x
            return total

        # Find the upper bound for binary search
        low = 0
        high = 1
        while compute_sum(high) <= k:
            high *= 2

        # Binary search to find the maximum num
        res = 0
        while low <= high:
            mid = (low + high) // 2
            current_sum = compute_sum(mid)
            if current_sum <= k:
                res = mid
                low = mid + 1
            else:
                high = mid - 1
        return res