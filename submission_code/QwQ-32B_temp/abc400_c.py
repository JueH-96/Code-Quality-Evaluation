import math

def count_good_integers(N):
    total = 0
    e = 1
    while True:
        denom = 2 ** e
        if denom > N:
            break
        value = N // denom
        s_max = math.isqrt(value)
        count = (s_max + 1) // 2
        total += count
        e += 1
    return total

N = int(input())
print(count_good_integers(N))