import sys
from collections import defaultdict

def f0(x):
    if x == 0:
        return 0
    while x % 2 == 0:
        x //= 2
    return x

def F_func(L):
    if not L:
        return 0
    if all(x == 0 for x in L):
        return 0
    L0 = [x for x in L if x % 2 == 0]
    L1 = [x for x in L if x % 2 == 1]
    L0_half = [x // 2 for x in L0]
    L1_half = [x // 2 for x in L1]
    n0 = len(L0)
    n1 = len(L1)
    s0 = sum(L0_half)
    s1 = sum(L1_half)
    part0 = F_func(L0_half)
    part1 = G_func(L1_half, 1)
    mixed = 4 * (n1 * s0 + n0 * s1) + 2 * n0 * n1
    return part0 + part1 + mixed

def G_func(L, c):
    if not L:
        return 0
    if all(x == 0 for x in L):
        return len(L) ** 2 * f0(c)
    L0 = [x for x in L if x % 2 == 0]
    L1 = [x for x in L if x % 2 == 1]
    n0 = len(L0)
    n1 = len(L1)
    s0 = sum([x // 2 for x in L0])
    s1 = sum([x // 2 for x in L1])
    if c % 2 == 0:
        part00 = G_func(L0, c // 2)
        part11 = G_func(L1, 1 + c // 2)
        part01 = 4 * (n1 * s0 + n0 * s1) + 2 * (1 + c) * n0 * n1
        return part00 + part11 + part01
    else:
        part00 = 4 * n0 * s0 + c * (n0 * n0)
        part11 = 4 * n1 * s1 + (2 + c) * (n1 * n1)
        d = (1 + c) // 2
        part01 = 2 * H_func(L0, L1, d)
        return part00 + part11 + part01

def H_func(A, B, d_val):
    if not A or not B:
        return 0
    if all(x == 0 for x in A) and all(x == 0 for x in B):
        return len(A) * len(B) * f0(d_val)
    A0 = [x for x in A if x % 2 == 0]
    A1 = [x for x in A if x % 2 == 1]
    B0 = [x for x in B if x % 2 == 0]
    B1 = [x for x in B if x % 2 == 1]
    n0A = len(A0)
    n1A = len(A1)
    n0B = len(B0)
    n1B = len(B1)
    s0A = sum([x // 2 for x in A0])
    s1A = sum([x // 2 for x in A1])
    s0B = sum([x // 2 for x in B0])
    s1B = sum([x // 2 for x in B1])
    if d_val % 2 == 0:
        part00 = H_func(A0, B0, d_val // 2)
        part11 = H_func(A1, B1, 1 + d_val // 2)
        part01 = n0A * n1B * (1 + d_val) + 2 * (n1B * s0A + n0A * s1B)
        part10 = n1A * n0B * (1 + d_val) + 2 * (n0B * s1A + n1A * s0B)
        return part00 + part11 + part01 + part10
    else:
        part00 = n0A * n0B * d_val + 2 * (n0A * s0B + n0B * s0A)
        part11 = n1A * n1B * (2 + d_val) + 2 * (n1A * s1B + n1B * s1A)
        d_half = (1 + d_val) // 2
        part01 = H_func(A0, B1, d_half)
        part10 = H_func(A1, B0, d_half)
        return part00 + part11 + part01 + part10

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1 + n]))
    groups = defaultdict(list)
    total_b = 0
    for a in A:
        k = 0
        x = a
        while x % 2 == 0:
            k += 1
            x //= 2
        groups[k].append(x)
        total_b += x
    T_same = 0
    for k, lst in groups.items():
        T_same += F_func(lst)
    cnt = {}
    F_sum = {}
    for k, lst in groups.items():
        cnt[k] = len(lst)
        F_sum[k] = sum(lst)
    keys = sorted(groups.keys())
    T_diff = 0
    for i in range(len(keys)):
        k1 = keys[i]
        for j in range(len(keys)):
            if i == j:
                continue
            k2 = keys[j]
            k_min = min(k1, k2)
            k_max = max(k1, k2)
            d = k_max - k_min
            term = F_sum[k_min] * cnt[k_max] + (2 ** d) * F_sum[k_max] * cnt[k_min]
            T_diff += term
    T = T_same + T_diff
    D = total_b
    ans = (T + D) // 2
    print(ans)

if __name__ == "__main__":
    main()