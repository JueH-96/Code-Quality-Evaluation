import math

def kth_smallest_divisible(N, M, K):
    lcm = N * M // math.gcd(N, M)
    count_per_lcm = lcm // N + lcm // M - 1
    full_cycles = K // count_per_lcm
    remaining = K % count_per_lcm

    result = full_cycles * lcm
    numbers = []
    for i in range(1, lcm // N + 1):
        if i * N % M != 0:
            numbers.append(i * N)
    for i in range(1, lcm // M + 1):
        if i * M % N != 0:
            numbers.append(i * M)
    numbers.sort()

    if remaining > 0:
        result += numbers[remaining - 1]

    return result

N, M, K = map(int, input().split())
print(kth_smallest_divisible(N, M, K))