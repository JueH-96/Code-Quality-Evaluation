class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        mod = 10**9 + 7
        # We'll need the modular inverse of 2 modulo mod.
        inv2 = (mod + 1) // 2  # equivalent to pow(2, mod-2, mod)

        # Explanation:
        # Consider the set of positive integers smaller than target.
        # For any two distinct numbers a and b with a + b == target, they form a forbidden pair.
        # Notice that for any a in [1, target-1], its complement is target - a.
        # We can group the numbers into pairs: (a, target - a) for a in [1, floor((target-1)/2)].
        #   - If target is odd, each such pair is distinct, and we can choose at most one number per pair.
        #     Thus the maximum safe count from below target is (target-1)//2.
        #   - If target is even, then besides the pairs (a, target - a) for a in [1, target//2 - 1],
        #     there is a “middle” number target/2. Since picking it once is allowed,
        #     the safe count becomes target//2.
        #
        # Therefore:
        if target % 2 == 0:
            allowed_low = target // 2
        else:
            allowed_low = (target - 1) // 2

        # Strategy:
        # 1. If n is small enough, all chosen numbers can come from the "allowed" smaller numbers.
        #    In that case, the best (minimum sum) beautiful array is the first n positive integers.
        # 2. Otherwise, we pick all allowed numbers from below target (which are the first allowed_low numbers)
        #    and then we take as many numbers from target upward as needed.
        #
        # Case 1: Use only numbers smaller than target.
        if n <= allowed_low:
            return (n % mod) * ((n + 1) % mod) % mod * inv2 % mod

        # Case 2: Use all allowed numbers from below target.
        # The sum of numbers from 1 to allowed_low is:
        low_sum = (allowed_low % mod) * ((allowed_low + 1) % mod) % mod * inv2 % mod

        # The remaining numbers:
        extra = n - allowed_low
        # They are chosen consecutively starting from target:
        # target, target+1, target+2, ..., target + extra - 1.
        # Their sum is an arithmetic progression:
        #   extra * (first + last) / 2 = extra * (target + (target + extra - 1)) // 2
        extra_sum = (extra % mod) * ((2 * (target % mod) + extra - 1) % mod) % mod * inv2 % mod

        return (low_sum + extra_sum) % mod


# Optional: For quick testing of the provided examples.
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumPossibleSum(2, 3))  # Expected output: 4
    print(sol.minimumPossibleSum(3, 3))  # Expected output: 8
    print(sol.minimumPossibleSum(1, 1))  # Expected output: 1