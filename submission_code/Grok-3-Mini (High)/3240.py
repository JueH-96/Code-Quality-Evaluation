class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def compute_sum(num):
            if num == 0:
                return 0
            max_p = num.bit_length() - 1
            total = 0
            p = x - 1
            while p <= max_p:
                cycle_len = 1 << (p + 1)
                cycle_set_count = 1 << p
                full_cycles = num // cycle_len
                set_in_full = full_cycles * cycle_set_count
                rem = num % cycle_len
                if rem >= (1 << p):
                    set_in_rem = rem - (1 << p) + 1
                else:
                    set_in_rem = 0
                bit_count = set_in_full + set_in_rem
                total += bit_count
                p += x
            return total
        
        # Binary search for the largest num such that sum <= k
        low = 1
        high = 1 << 60  # 2^60, a large upper bound
        while low <= high:
            mid = (low + high) // 2
            if compute_sum(mid) <= k:
                low = mid + 1
            else:
                high = mid - 1
        # After binary search, high is the largest num with sum <= k
        return high