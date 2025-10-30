class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        # Explanation:
        # We need to find the maximum integer num such that the sum of prices
        # for all integers from 1 to num does not exceed k.
        # The "price" for a given number is defined by its 1-indexed binary representation,
        # counting only those positions i (starting at the right, with i=1 for the LSB)
        # that are multiples of x and where the bit is set.
        #
        # Notice that for each bit position i (where i % x == 0) in a number,
        # the count of numbers in [1, n] having that bit set can be computed without iterating through each number.
        #
        # For a given position i, where the bit value is 2^(i-1) and the period = 2^i,
        # the number of set bits among the numbers 1 to n is:
        #    floor(n / (2^i)) * 2^(i-1) + max(0, (n % 2^i) - 2^(i-1) + 1)
        #
        # Let’s denote pos = j * x (i.e. positions that are multiples of x).
        # Then the overall function f(n) which sums the "prices" for numbers from 1 to n is:
        #    f(n) = sum_{j>=1 such that 2^(j*x - 1) <= n} 
        #             [  (n // (2^(j*x)))*2^(j*x-1) + max(0, (n % (2^(j*x))) - 2^(j*x-1) + 1 ) ]
        #
        # Since f(n) is non-decreasing, we can use binary search to find the maximum num 
        # (i.e., maximum n) such that f(n) <= k.
        #
        # Given constraints (k up to 1e15), an upper bound of 10^18 is safe.
        
        def f(n: int, x: int) -> int:
            total = 0
            j = 1
            # Each j gives us a bit position pos = j*x (1-indexed)
            while True:
                pos = j * x  # the bit position we're checking
                base = 1 << (pos - 1)  # bit value = 2^(pos-1)
                period = 1 << pos      # period = 2^pos
                if base > n:  # If even the smallest number with that bit set is > n, break
                    break
                full_cycles = n // period
                count = full_cycles * base
                remainder = n % period
                extra = remainder - base + 1
                if extra < 0:
                    extra = 0
                count += extra
                total += count
                j += 1
            return total
        
        lo, hi = 0, 10**18
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if f(mid, x) <= k:
                lo = mid
            else:
                hi = mid - 1
        return lo

# For local testing (optional):
def solve():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    k = int(data[0])
    x = int(data[1])
    sol = Solution()
    sys.stdout.write(str(sol.findMaximumNumber(k, x)))

if __name__ == "__main__":
    solve()