import sys

MOD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def count_set_bits(n):
    count = 0
    while n > 0:
        count += n & 1
        n >>= 1
    return count

def get_exponents(n):
    exponents = []
    for i in range(30, -1, -1):
        if (n >> i) & 1:
            exponents.append(i)
    return exponents

def sum_exponents_upto(x):
    if x == 0:
        return 0
    # Binary search to find m where S(m) <= x < S(m+1)
    low, high = 0, x
    m = 0
    while low <= high:
        mid = (low + high) // 2
        s_mid = sum_exponents_upto_m(mid)
        if s_mid <= x:
            m = mid
            low = mid + 1
        else:
            high = mid - 1
    sum_m = sum_exponents_upto_m(m)
    s_m = sum_exponents_upto_m(m)
    k = x - s_m
    if k <= 0:
        return sum_m
    m_plus_1 = m + 1
    exponents = get_exponents(m_plus_1)
    sum_partial = sum(exponents[:min(k, len(exponents))])
    return sum_m + sum_partial

def sum_exponents_upto_m(m):
    total = 0
    for i in range(1, m + 1):
        count = count_set_bits(i)
        total += count
    return total

def find_m(x):
    low, high = 0, x
    while low <= high:
        mid = (low + high) // 2
        s_mid = sum_exponents_upto_m(mid)
        if s_mid <= x:
            low = mid + 1
        else:
            high = mid - 1
    return high

def solve():
    q = int(sys.stdin.readline())
    for _ in range(q):
        a, b = map(int, sys.stdin.readline().split())
        if a == 0:
            a = 1
        if a > b:
            print(1)
            continue
        # Compute sum_exponents_upto(b)
        S_b = sum_exponents_upto(b)
        # Compute sum_exponents_upto(a-1)
        S_a_minus_1 = sum_exponents_upto(a-1)
        S = S_b - S_a_minus_1
        mod = MOD[_]
        print(pow(2, S, mod))

if __name__ == "__main__":
    solve()