class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        # Helper function: for a given N, compute the total "price"
        # defined as the sum over num=1..N of the number of set bits at positions i (1-indexed)
        # where i % x == 0.
        # For each such bit position i, we want to add how many numbers from 1 to N have that bit set.
        #
        # To compute how many numbers in range [1, N] have the j-th bit set, where j is 1-indexed,
        # we can use a standard pattern:
        # Let power = 2**j_minus1 and next_power = 2**j.
        # then countSet = (N // next_power) * power + max(0, (N % next_power) - power + 1)
        #
        # We only sum over j that are multiples of x: i.e. j = x, 2x, 3x, ... up to j_max,
        # where 2^(j-1) <= N. We can safely iterate over j until 2^(j-1) > N.
        def total_price(N: int) -> int:
            result = 0
            j = x  # start at first position that is a multiple of x
            while True:
                power = 1 << (j - 1)  # 2^(j-1)
                if power > N:
                    break
                next_power = power << 1  # 2^j
                # Count how many numbers in 1..N have the j-th bit set.
                count = (N // next_power) * power + max(0, (N % next_power) - power + 1)
                result += count
                j += x  # next multiple of x
            return result

        # We need to find the maximum N that satisfies total_price(N) <= k.
        # We can do this with binary search.
        lo, hi = 1, 10**18  # hi chosen as an upper bound (should be safe given the constraints)
        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            tp = total_price(mid)
            if tp <= k:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans