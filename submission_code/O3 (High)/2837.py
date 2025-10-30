class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        """
        We need k (number of operations) and powers 2^{i_j} (0 ≤ i_j ≤ 60) such that

               num1 - k*num2 = Σ 2^{i_j}   (j = 1 … k)

        Let  T = num1 - k*num2.
        The right-hand side is a sum of k (not necessarily distinct) powers of two,
        therefore:

            1. T must be non–negative.
            2. It has at least `popcount(T)` summands (its set bits)          ⇒ k ≥ popcount(T)
            3. It has at most  T  summands (all 1s)                           ⇒ k ≤ T

        We iterate over k from 1 to 60 (the maximal distinct power index we are
        allowed to use).  The first k that satisfies

                T = num1 - k*num2  ≥ k   and   popcount(T) ≤ k

        is the minimum number of operations; if no such k exists we return –1.
        """
        for k in range(1, 61):                 # try every possible number of operations
            t = num1 - k * num2                # required sum of powers of two
            if t < k:                          # condition 1  (also guarantees t >= 0)
                continue
            # condition 2: number of set bits not greater than k
            if bin(t).count('1') <= k:
                return k
        return -1