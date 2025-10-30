import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    constraints = []
    R_set = set()
    C_set = set()
    for _ in range(M):
        x, y, c = sys.stdin.readline().split()
        x = int(x)
        y = int(y)
        constraints.append((x, y, c))
        R_set.add(x)
        C_set.add(y)

    # 1. Compute L_X, R_X, L'_Y, R'_Y ranges
    # Default ranges [0, N]
    L = {r: 0 for r in R_set}
    R = {r: N for r in R_set}
    L_prime = {c: 0 for c in C_set}
    R_prime = {c: N for c in C_set}

    for x, y, c in constraints:
        if c == 'B':
            L[x] = max(L[x], y)
            L_prime[y] = max(L_prime[y], x)
        else: # c == 'W'
            R[x] = min(R[x], y - 1)
            R_prime[y] = min(R_prime[y], x - 1)

    # 2. Check L_X <= R_X, L'_Y <= R'_Y
    for r in R_set:
        if L[r] > R[r]:
            print("No")
            return
    for c in C_set:
        if L_prime[c] > R_prime[c]:
            print("No")
            return

    # 3. Check white cell condition
    for x, y, c in constraints:
        if c == 'W':
            if y <= L[x] and x <= L_prime[y]:
                 print("No")
                 return

    # Add boundaries for R and C indices
    R_indices = sorted(list(R_set) + [0, N + 1])
    C_indices = sorted(list(C_set) + [0, N + 1])
    
    # Bounds for i sequence: L_r, R_r (for r in R_set), L_0, R_0, L_{N+1}, R_{N+1}
    # i_0=N, i_{N+1}=0. L_0=0, R_0=N. L_{N+1}=0, R_{N+1}=0.
    L_with_bounds = {0: 0, N + 1: 0}
    L_with_bounds.update(L)
    R_with_bounds = {0: N, N + 1: 0} 
    R_with_bounds.update(R)

    # Bounds for j sequence: L'_c, R'_c (for c in C_set), L'_0, R'_0, L'_{N+1}, R'_{N+1}
    # j_0=N, j_{N+1}=0. L'_0=0, R'_0=N. L'_{N+1}=0, R'_{N+1}=0.
    L_prime_with_bounds = {0: 0, N + 1: 0}
    L_prime_with_bounds.update(L_prime)
    R_prime_with_bounds = {0: N, N + 1: 0}
    R_prime_with_bounds.update(R_prime)

    # 4. Check R_{r_a} >= L_{r_{a+1}} for consecutive r_a, r_{a+1} in sorted R_indices
    # This checks if a non-increasing sequence i exists satisfying L/R constraints at R_set
    for i in range(len(R_indices) - 1):
        r_a = R_indices[i]
        r_b = R_indices[i+1]
        if R_with_bounds[r_a] < L_with_bounds[r_b]:
            print("No")
            return

    # 5. Check R'_{c_a} >= L'_{c_{a+1}} for consecutive c_a, c_{c+1} in sorted C_indices
    # This checks if a non-increasing sequence j exists satisfying L'/R' constraints at C_set
    for i in range(len(C_indices) - 1):
        c_a = C_indices[i]
        c_b = C_indices[i+1]
        if R_prime_with_bounds[c_a] < L_prime_with_bounds[c_b]:
            print("No")
            return

    # 6. Check L'_Y <= max({r | i^{lower}_r >= Y}) for Y in C_set
    # max({r | i^{lower}_r >= Y}) = max({x in R_indices | L_x >= Y})
    
    Lx_pairs = [(L_with_bounds[x], x) for x in R_indices]
    Lx_pairs.sort(key=lambda item: item[0], reverse=True)

    sorted_C_set = sorted(list(C_set), reverse=True)
    
    lx_iter = iter(Lx_pairs)
    current_max_x = 0 # max({x in R_indices | L_x >= y})
    next_lx = next(lx_iter, None)
    
    for y in sorted_C_set:
        while next_lx is not None and next_lx[0] >= y:
            current_max_x = max(current_max_x, next_lx[1])
            next_lx = next(lx_iter, None)
        
        # We need L'_y <= current_max_x
        if L_prime[y] > current_max_x:
            print("No")
            return

    # 7. Check R'_Y >= max({r | i^{upper}_r >= Y}) for Y in C_set
    # max({r | i^{upper}_r >= Y}) = min({x in R_indices | R_x < Y} U {N+1}) - 1

    Rx_pairs = [(R_with_bounds[x], x) for x in R_indices]
    Rx_pairs.sort(key=lambda item: item[0]) # Sort by R_x ascending

    sorted_C_set_asc = sorted(list(C_set))
    
    rx_iter = iter(Rx_pairs)
    current_min_x = N + 1 # min({x in R_indices | R_x < y} U {N+1})
    next_rx = next(rx_iter, None)

    # To handle R_x < Y check for Y=0 correctly (where R_x < 0 is never true)
    # R_with_bounds includes R_{N+1}=0. We need R_{N+1}=-1 for R_x < Y logic
    # Re-add N+1 point with R_x = -1
    Rx_pairs_adjusted = [(R_with_bounds.get(x, R[x]), x) for x in R_indices[:-1]] # Exclude the original N+1 point
    Rx_pairs_adjusted.append((-1, N+1)) # Add N+1 with R_{N+1}=-1
    Rx_pairs_adjusted.sort(key=lambda item: item[0])

    rx_iter_adjusted = iter(Rx_pairs_adjusted)
    current_min_x = N + 1 # min({x in R_indices | R_x < y} U {N+1})
    next_rx_adjusted = next(rx_iter_adjusted, None)

    for y in sorted_C_set_asc:
        # Advance rx_ptr while R_x < y
        while next_rx_adjusted is not None and next_rx_adjusted[0] < y:
            current_min_x = min(current_min_x, next_rx_adjusted[1])
            next_rx_adjusted = next(rx_iter_adjusted, None)

        # Current_min_x is min({x in R_indices | R_x < y} U {N+1})
        # We need R'_y >= current_min_x - 1
        if R_prime[y] < current_min_x - 1:
            print("No")
            return

    print("Yes")

solve()