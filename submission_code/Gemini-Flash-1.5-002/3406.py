class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        mod = 10**9 + 7
        n = zero + one
        dp = {}

        def count_stable_arrays(zeros_left, ones_left, last_zero, last_one):
            if zeros_left == 0 and ones_left == 0:
                return 1
            if (zeros_left, ones_left, last_zero, last_one) in dp:
                return dp[(zeros_left, ones_left, last_zero, last_one)]

            count = 0
            if zeros_left > 0:
                if last_one + 1 <= limit:
                    count = (count + count_stable_arrays(zeros_left - 1, ones_left, last_zero + 1, 0)) % mod
            if ones_left > 0:
                if last_zero + 1 <= limit:
                    count = (count + count_stable_arrays(zeros_left, ones_left - 1, 0, last_one + 1)) % mod

            dp[(zeros_left, ones_left, last_zero, last_one)] = count
            return count

        return count_stable_arrays(zero, one, 0, 0)