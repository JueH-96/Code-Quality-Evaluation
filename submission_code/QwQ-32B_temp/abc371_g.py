import sys
import math

def extended_gcd(a, b):
    if b == 0:
        return (a, 1, 0)
    else:
        g, x, y = extended_gcd(b, a % b)
        return (g, y, x - (a // b) * y)

def crt(a1, m1, a2, m2):
    g, p, q = extended_gcd(m1, m2)
    if (a2 - a1) % g != 0:
        return None
    lcm = m1 // g * m2
    x0 = (a1 + (a2 - a1) // g * p * m1) % lcm
    return (x0, lcm)

def main():
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))

    visited = [False] * (N + 1)
    cycles = []
    for i in range(1, N + 1):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = P[current - 1]
            cycles.append(cycle)

    cycle_info = []
    for cycle in cycles:
        elements = [A[node - 1] for node in cycle]
        min_val = min(elements)
        s_i = elements.index(min_val)
        cycle_info.append((cycle, elements, s_i))

    sorted_cycles = sorted(cycle_info, key=lambda x: x[0][0])

    current_a = 0
    current_m = 1

    for (cycle, elements, desired_s) in sorted_cycles:
        k_i = len(cycle)
        s_i = desired_s

        g = math.gcd(current_m, k_i)
        if (s_i - current_a) % g != 0:
            r = current_a % g
            best_shift = None
            best_sequence = None
            for s_candidate in range(k_i):
                if s_candidate % g == r % g:
                    shifted = elements[s_candidate:] + elements[:s_candidate]
                    if best_sequence is None or shifted < best_sequence:
                        best_sequence = shifted
                        best_shift = s_candidate
            s_i_prime = best_shift
            new_congruence = crt(current_a, current_m, s_i_prime, k_i)
            if new_congruence:
                current_a, current_m = new_congruence
        else:
            new_congruence = crt(current_a, current_m, s_i, k_i)
            if new_congruence:
                current_a, current_m = new_congruence

    t = current_a % current_m

    result = [0] * N
    for (cycle, elements, desired_s) in cycle_info:
        k_i = len(cycle)
        shift = t % k_i
        shifted = elements[shift:] + elements[:shift]
        for idx in range(k_i):
            node = cycle[idx]
            result[node - 1] = shifted[idx]

    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()