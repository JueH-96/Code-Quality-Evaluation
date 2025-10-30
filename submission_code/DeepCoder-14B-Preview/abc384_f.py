import sys
from collections import defaultdict

def compute_sum(mod, A):
    freq = defaultdict(int)
    sum_r = defaultdict(int)
    for a in A:
        r = a % mod
        freq[r] += 1
        sum_r[r] += a
    sum_all = 0
    for a in A:
        r = a % mod
        s = (-r) % mod  # Equivalent to (mod - r) % mod
        sum_all += a * freq[s] + sum_r.get(s, 0)
    sum_i_j_eq = 0
    for a in A:
        if (2 * a) % mod == 0:
            sum_i_j_eq += 2 * a
    sum_i_leq_j = (sum_all + sum_i_j_eq) // 2
    return sum_i_leq_j

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    total = 0
    for k in range(31):
        mod = 1 << k
        sum_k = compute_sum(mod, A)
        if k < 30:
            mod_next = 1 << (k + 1)
            sum_kp1 = compute_sum(mod_next, A)
        else:
            sum_kp1 = 0
        contribution = (sum_k - sum_kp1) // mod
        total += contribution
    print(total)

if __name__ == '__main__':
    main()