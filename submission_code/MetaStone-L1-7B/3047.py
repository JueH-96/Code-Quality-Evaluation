import sys
import math
from collections import defaultdict

def maximum_element_sum(nums: List[int]) -> int:
    if not nums:
        return 0

    max_num = max(nums)
    max_spf = int(math.isqrt(max_num)) + 1
    spf = list(range(max_spf + 1))
    for i in range(2, int(math.isqrt(max_spf)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_spf + 1, i):
                if spf[j] == j:
                    spf[j] = i

    square_free = defaultdict(int)
    for x in nums:
        if x == 0:
            square_free[0] += x
            continue

        factors = {}
        temp = x
        while temp != 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                count += 1
                temp = temp // p
            if count % 2 != 0:
                if p in factors:
                    factors[p] += 1
                else:
                    factors[p] = 1
        s = 1
        for p in factors:
            s *= p
        square_free[s] += x

    max_sum = 0
    for key in square_free:
        current_sum = square_free[key]
        if current_sum > max_sum:
            max_sum = current_sum
    return max_sum