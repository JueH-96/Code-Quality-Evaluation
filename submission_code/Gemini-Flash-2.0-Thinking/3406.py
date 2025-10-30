class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        mod = 10**9 + 7
        dp_zero = {}
        dp_one = {}

        def solve_zero(z, o):
            if (z, o) in dp_zero:
                return dp_zero[(z, o)]
            if z == 0 and o > 0:
                return 0
            if z > 0 and o == 0 and z <= limit:
                return 1
            if z > 0 and o == 0 and z > limit:
                return 0

            count = 0
            for k in range(1, limit + 1):
                if z - k >= 0:
                    count = (count + solve_one(z - k, o)) % mod
            dp_zero[(z, o)] = count
            return count

        def solve_one(z, o):
            if (z, o) in dp_one:
                return dp_one[(z, o)]
            if o == 0 and z > 0:
                return 0
            if o > 0 and z == 0 and o <= limit:
                return 1
            if o > 0 and z == 0 and o > limit:
                return 0

            count = 0
            for k in range(1, limit + 1):
                if o - k >= 0:
                    count = (count + solve_zero(z, o - k)) % mod
            dp_one[(z, o)] = count
            return count

        return (solve_zero(zero, one) + solve_one(zero, one)) % mod