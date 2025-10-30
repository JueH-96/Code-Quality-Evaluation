import sys
import math
from math import gcd

def extended_gcd(a, b):
    if b == 0:
        return (a, 1, 0)
    else:
        g, x, y = extended_gcd(b, a % b)
        return (g, y, x - (a // b) * x)

def crt(a1, m1, a2, m2):
    g, x, y = extended_gcd(m1, m2)
    if (a2 - a1) % g != 0:
        return (0, -1)
    lcm = m1 // g * m2
    m1_prime = m1 // g
    m2_prime = m2 // g
    diff = (a2 - a1) // g
    g_inv, inv, _ = extended_gcd(m1_prime, m2_prime)
    assert g_inv == 1
    k = (diff * inv) % m2_prime
    x = a1 + k * m1
    return (x, lcm)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    P = list(map(lambda x: int(x)-1, input[ptr:ptr+N]))
    ptr += N
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    visited = [False] * N
    cycles = []
    for i in range(N):
        if not visited[i]:
            current = i
            new_cycle = []
            while not visited[current]:
                visited[current] = True
                new_cycle.append(current)
                current = P[current]
            cycles.append(new_cycle)
    cycles.sort(key=lambda x: min(x))

    cycle_info = []
    for cycle in cycles:
        sorted_positions = sorted(cycle)
        pos_to_index = {p: idx for idx, p in enumerate(cycle)}
        cycle_info.append( (cycle, sorted_positions, pos_to_index) )

    a = 0
    M = 1
    rotations = []

    for info in cycle_info:
        cycle, sorted_positions, pos_to_index = info
        m = len(cycle)
        g = math.gcd(M, m)
        c = a % g

        possible_rs = []
        r_start = c
        step = g
        while r_start < m:
            possible_rs.append(r_start)
            r_start += step

        min_sequence = None
        best_r = 0
        for r in possible_rs:
            sequence = []
            for p in sorted_positions:
                idx_in_cycle = pos_to_index[p]
                new_idx = (idx_in_cycle + r) % m
                sequence.append(A[cycle[new_idx]])
            if (min_sequence is None) or (sequence < min_sequence):
                min_sequence = sequence
                best_r = r
        rotations.append(best_r)
        new_a, new_M = crt(a, M, best_r, m)
        a = new_a
        M = new_M

    res = A.copy()
    for i in range(len(cycles)):
        cycle = cycles[i]
        r = rotations[i]
        m = len(cycle)
        original_values = [A[p] for p in cycle]
        rotated_values = original_values[r:] + original_values[:r]
        for idx_in_cycle in range(m):
            res[cycle[idx_in_cycle]] = rotated_values[idx_in_cycle]
    print(' '.join(map(str, res)))

if __name__ == "__main__":
    main()