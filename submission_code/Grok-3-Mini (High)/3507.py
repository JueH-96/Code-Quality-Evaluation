import math

class Solution:
    def nonSpecialCount(self, l: int, r: int) -> int:
        total = r - l + 1
        sqrt_l = math.sqrt(l)
        a_ceil = math.ceil(sqrt_l)
        a_int = int(a_ceil)
        sqrt_r = math.sqrt(r)
        b_floor = math.floor(sqrt_r)
        b_int = int(b_floor)
        low = max(a_int, 2)
        high = b_int
        if low > high:
            special_count = 0
        else:
            max_p = 31622  # Maximum possible floor(sqrt(10^9))
            is_prime = [True] * (max_p + 1)
            is_prime[0] = False
            is_prime[1] = False
            sieve_limit = int(math.sqrt(max_p))
            for i in range(2, sieve_limit + 1):
                if is_prime[i]:
                    for j in range(i * i, max_p + 1, i):
                        is_prime[j] = False
            special_count = sum(1 for p in range(low, high + 1) if is_prime[p])
        non_special = total - special_count
        return non_special