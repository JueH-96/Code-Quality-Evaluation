MOD = 10**9 + 7
inv2 = (MOD+1)//2  # Modular inverse of 2 modulo MOD

class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        # Helper function: sum of first x natural numbers modulo MOD.
        def sum_first(x):
            # (x*(x+1)/2) mod MOD using modular inverse of 2.
            return (x % MOD) * ((x+1) % MOD) % MOD * inv2 % MOD
        
        # Helper: Sum of an arithmetic sequence of r numbers starting at A with common difference 1:
        # Sum = r*A + (r-1)*r/2.
        def arithmetic_sum(A, r):
            return ((r % MOD) * (A % MOD) % MOD + ((r-1) % MOD * (r % MOD) % MOD * inv2) % MOD) % MOD

        # There is a natural partition of the candidates into two zones:
        #  A: Numbers < target.
        #  B: Numbers >= target.
        #
        # In A, note that for any positive x, if x < target then also target - x < target.
        # Thus numbers in A appear in complementary pairs (x, target-x) for x from 1 to target-1.
        # (When target is even, there is one "middle" number: target//2, because target//2 + target//2 = target.
        #  Choosing it once is safe, but you cannot choose it twice.)
        #
        # To minimize the total sum while meeting all conditions, our idea is:
        #  1. First pick candidates from A (the “low” numbers) as long as we can select them
        #     without picking both numbers from a complementary pair.
        #  2. In an optimal choice we choose the smaller number in each pair.
        #  3. If n is larger than what we can safely pick from A (call that count m),
        #     then we take all the “optimal” numbers from A and complete the selection using
        #     the smallest numbers from B (which are consecutive starting at target
        #     and are always safe because any sum will be at least 2*target).
        
        # Two cases depending on parity of target:
        if target & 1:  # target is odd.
            # All numbers in A are in complementary pairs:
            # The pairs are (x, target-x) for x from 1 to (target-1),
            # and the optimal (smaller) choices come from x in [1, (target-1)//2].
            m = (target - 1) // 2  # maximum picks from A
            if n <= m:
                # We can pick the n smallest numbers from the available set in A: {1, 2, ..., m}.
                # Their sum is just 1+2+...+n.
                return sum_first(n)
            else:
                # We use all m numbers from A.
                A_sum = sum_first(m)
                # The remaining numbers must come from B.
                r = n - m
                # B: the r smallest numbers starting at target: target, target+1, ..., target + r - 1.
                B_sum = arithmetic_sum(target, r)
                return (A_sum + B_sum) % MOD
        else:
            # target is even.
            # In A we have pairs (x, target-x) for x from 1 to target-1 where x < target//2,
            # plus a unique middle candidate: target//2.
            #
            # The optimal selection from A is:
            #   • From each pair (x, target-x) with x from 1 to target//2 - 1, choose x.
            #   • Also, choose target//2.
            #
            # So the sorted list of available A candidates (in increasing order) is:
            #   [1, 2, ..., target//2 - 1, target//2].
            # Let k = target//2 - 1 so that m (maximum count from A) = k + 1 = target//2.
            m = target // 2
            k = m - 1
            if n <= k:
                # We’re taking only from the first part: [1, 2, ..., n].
                return sum_first(n)
            elif n == m:
                # Must take all optimal elements from A: the first k numbers plus target//2.
                return (sum_first(k) + m) % MOD
            else:
                # Use full A and select the remaining from B.
                A_sum = (sum_first(k) + m) % MOD
                r = n - m
                B_sum = arithmetic_sum(target, r)
                return (A_sum + B_sum) % MOD

# Below is a simple test harness.
if __name__ == '__main__':
    sol = Solution()
    # Example 1: n = 2, target = 3. The beautiful array [1,3] has sum 4.
    print(sol.minimumPossibleSum(2, 3))  # Expected output: 4

    # Example 2: n = 3, target = 3. A choice is [1,3,4] with sum 8.
    print(sol.minimumPossibleSum(3, 3))  # Expected output: 8

    # Example 3: n = 1, target = 1. The beautiful array [1] has sum 1.
    print(sol.minimumPossibleSum(1, 1))  # Expected output: 1

    # Additional tests:
    # For target = 4 (even) and n = 2, optimal array is [1,2] with sum 3.
    print(sol.minimumPossibleSum(2, 4))  # Expected output: 3

    # For target = 4 and n = 3, optimal array is [1,2,4] with sum 7.
    print(sol.minimumPossibleSum(3, 4))  # Expected output: 7

    # For target = 10 (even) and n = 7:
    # Optimal A: [1,2,3,4,5]  (with 5 = target//2), then B: [10, 11]
    # Sum = (1+2+3+4+5) + (10+11) = 15 + 21 = 36.
    print(sol.minimumPossibleSum(7, 10))  # Expected output: 36