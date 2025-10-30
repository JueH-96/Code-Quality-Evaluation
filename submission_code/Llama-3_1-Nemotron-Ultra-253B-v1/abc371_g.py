import math

def crt(a1, m1, a2, m2):
    d = math.gcd(m1, m2)
    if (a1 - a2) % d != 0:
        return None
    m1_dash = m1 // d
    m2_dash = m2 // d
    a_dash = (a2 - a1) // d
    try:
        inv = pow(m1_dash, -1, m2_dash)
    except ValueError:
        return None
    k = (a_dash * inv) % m2_dash
    x = a1 + k * m1
    m = m1 // d * m2
    return x % m, m

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    P = list(map(lambda x: int(x)-1, input[ptr:ptr+N]))
    ptr +=N
    A = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    
    visited = [False]*N
    cycles = []
    for i in range(N):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = P[j]
            cycles.append(cycle)
    
    cycles.sort(key=lambda x: min(x))
    
    preprocessed = []
    for cycle in cycles:
        sorted_pos = sorted(cycle)
        indices = [cycle.index(p) for p in sorted_pos]
        elements = [A[p] for p in cycle]
        preprocessed.append( (cycle, sorted_pos, indices, elements) )
    
    a = 0
    m = 1
    for cycle_info in preprocessed:
        cycle, sorted_pos, indices, elements = cycle_info
        L = len(cycle)
        d = math.gcd(m, L)
        r_rem = a % d
        allowed_rs = []
        r = r_rem
        while r < L:
            allowed_rs.append(r)
            r += d
        
        best_r = None
        best_elem = None
        for r in allowed_rs:
            current = [ elements[(i + r) % L] for i in indices ]
            if best_elem is None or current < best_elem:
                best_elem = current
                best_r = r
            elif current == best_elem and r < best_r:
                best_r = r
        
        new_a, new_m = crt(a, m, best_r, L)
        a, m = new_a, new_m
    
    result = [0]*N
    for cycle_info in preprocessed:
        cycle, _, _, elements = cycle_info
        L = len(cycle)
        r = a % L
        rotated = elements[r:] + elements[:r]
        for i in range(L):
            result[cycle[i]] = rotated[i]
    
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()