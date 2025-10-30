import sys
import math

def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = extended_gcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = extended_gcd(a, m)
    if g != 1:
        return None  # Inverse doesn't exist
    else:
        return x % m

def solve_congruence(a1, m1, a2, m2):
    g, x, y = extended_gcd(m1, m2)
    if (a2 - a1) % g != 0:
        return (a1, m1)  # Placeholder for no solution case
    lcm = m1 // g * m2
    m1_div = m1 // g
    m2_div = m2 // g
    a_diff = (a2 - a1) // g
    inv_m1_div = modinv(m1_div, m2_div)
    if inv_m1_div is None:
        return (a1, m1)
    k = (a_diff * inv_m1_div) % m2_div
    x0 = a1 + m1 * k
    x0 %= lcm
    return (x0, lcm)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    P = list(map(lambda x: int(x) - 1, input[ptr:ptr + N]))
    ptr += N
    A = list(map(lambda x: int(x) - 1, input[ptr:ptr + N]))
    ptr += N
    
    visited = [False] * N
    cycle_val = [[] for _ in range(N)]
    cycle_idx = [0] * N
    
    for i in range(N):
        if not visited[i]:
            current = i
            cycle = []
            while True:
                visited[current] = True
                cycle.append(current)
                current = P[current]
                if current == i:
                    break
            values = [A[pos] for pos in cycle]
            for idx, pos in enumerate(cycle):
                cycle_val[pos] = values
                cycle_idx[pos] = idx
    
    m = 1
    a = 0
    for i in range(N):
        c = len(cycle_val[i])
        values = cycle_val[i]
        j = cycle_idx[i]
        if c == 0:
            continue
        g = math.gcd(m, c)
        base_r = a % c
        possible_r = []
        num_s = c // g
        for s in range(num_s):
            r = (base_r + s * g) % c
            possible_r.append(r)
        min_val = None
        min_rs = []
        for r in possible_r:
            idx_in_cycle = (j + r) % c
            val = values[idx_in_cycle]
            if min_val is None or val < min_val:
                min_val = val
                min_rs = [r]
            elif val == min_val:
                min_rs.append(r)
        new_mod = m * c // g
        possible_solutions = []
        for r in min_rs:
            sol, lcm_val = solve_congruence(a, m, r, c)
            possible_solutions.append(sol)
        if possible_solutions:
            minimal = min((sol % new_mod for sol in possible_solutions))
        else:
            minimal = a
        m = new_mod
        a = minimal % m
    
    result = []
    for i in range(N):
        c = len(cycle_val[i])
        j = cycle_idx[i]
        idx_in_cycle = (j + a) % c
        val = cycle_val[i][idx_in_cycle]
        result.append(val + 1)
    
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()