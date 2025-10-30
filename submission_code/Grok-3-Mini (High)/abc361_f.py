import math

def integer_root(N, k):
    low = 1
    high = N
    while low <= high:
        mid = (low + high) // 2
        if mid ** k <= N:
            low = mid + 1
        else:
            high = mid - 1
    return high

def floor_log_base(m, N):
    k = 0
    power = 1
    while True:
        next_power = power * m
        if next_power > N:
            return k
        power = next_power
        k += 1

def perfect_powers_up_to(N):
    powers = set()
    if N >= 1:
        powers.add(1)
    for b in range(2, 64):
        if 2 ** b > N:
            break
        max_a = integer_root(N, b)
        for a in range(2, max_a + 1):
            power = a ** b
            powers.add(power)
    return powers

def count_perfect_powers(N):
    if N == 1:
        return 1
    S = integer_root(N, 2)
    K_max = N.bit_length() - 1
    sum_all_floor_log = 0
    for k in range(2, K_max + 1):
        floor_N1k = integer_root(N, k)
        floor_N1k1 = integer_root(N, k + 1)
        L = max(2, floor_N1k1 + 1)
        U = min(S, floor_N1k)
        if L <= U:
            num_m = U - L + 1
            sum_all_floor_log += k * num_m
    num_m_all = S - 1
    sum_all_minus_one = sum_all_floor_log - num_m_all
    P = perfect_powers_up_to(S)
    sum_perfect = 0
    for m in P:
        if m >= 2:
            log_k = floor_log_base(m, N)
            sum_perfect += log_k - 1
    sum_not_perfect = sum_all_minus_one - sum_perfect
    result = 1 + sum_not_perfect
    return result

# Read input from stdin
N = int(input())
# Compute and output the result
print(count_perfect_powers(N))