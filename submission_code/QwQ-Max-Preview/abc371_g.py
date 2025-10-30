def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    P = list(map(int, input[idx:idx+N]))
    idx += N
    A = list(map(int, input[idx:idx+N]))
    idx += N

    # Decompose P into cycles
    visited = [False] * (N + 1)  # 1-based
    cycles = []
    for i in range(1, N + 1):
        if not visited[i]:
            current = i
            cycle = []
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = P[current - 1]  # P is 1-based
            cycles.append(cycle)
    
    # For each cycle, compute the minimal rotation s_i
    congruences = []
    cycle_info = []
    for cycle in cycles:
        elements = [A[i-1] for i in cycle]
        min_val = min(elements)
        min_pos = elements.index(min_val)
        s_i = min_pos
        L_i = len(cycle)
        congruences.append((s_i, L_i))
        cycle_info.append((cycle, elements, s_i))
    
    # Solve the system of congruences
    a = 0
    m = 1
    has_solution = True
    for (s_i, L_i) in congruences:
        # Solve x ≡ a mod m
        # and x ≡ s_i mod L_i
        d = gcd(m, L_i)
        if (s_i - a) % d != 0:
            has_solution = False
            break
        # Find new_a and new_m
        lcm = m // d * L_i
        _gcd, x, y = extended_gcd(m, L_i)
        tmp = (s_i - a) // d * x % (L_i // d)
        new_a = a + m * tmp
        new_a %= lcm
        a = new_a
        m = lcm
    
    if has_solution:
        k = a
    else:
        # Find the earliest cycle (with the smallest index)
        min_idx = float('inf')
        earliest_cycle = None
        earliest_si = 0
        earliest_Li = 0
        for cycle, elements, s_i in cycle_info:
            current_min = min(cycle)
            if current_min < min_idx:
                min_idx = current_min
                earliest_cycle = cycle
                earliest_si = s_i
                earliest_Li = len(cycle)
        # Use the earliest cycle's s_i as k
        k = earliest_si
    
    # Generate the result array
    res = [0] * N
    for cycle, elements, s_i in cycle_info:
        L = len(cycle)
        s = k % L
        for i in range(L):
            original_pos = cycle[(i + s) % L]
            res[cycle[i] - 1] = elements[(i + s) % L]
    
    print(' '.join(map(str, res)))

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = extended_gcd(b % a, a)
        return (g, x - (b // a) * y, y)

if __name__ == '__main__':
    main()