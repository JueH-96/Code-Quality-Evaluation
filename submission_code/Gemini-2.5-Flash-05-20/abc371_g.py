import sys

# For faster I/O
input = sys.stdin.readline

def extended_gcd(a, b):
    """
    Returns (gcd, x, y) such that a*x + b*y = gcd(a, b).
    This is used to solve linear congruences in CRT.
    """
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def smallest_cyclic_shift_index(arr):
    """
    Finds the starting index of the lexicographically smallest cyclic shift of arr.
    Uses Duval's algorithm, which runs in O(N) time.
    """
    n = len(arr)
    if n == 0:
        return 0
    
    i, j, k = 0, 1, 0
    # i: current candidate for the starting index of the smallest shift
    # j: another candidate for the starting index of the smallest shift
    # k: number of characters matched so far between arr[i...i+k-1] and arr[j...j+k-1]

    while i < n and j < n and k < n:
        val_i_k = arr[(i + k) % n]
        val_j_k = arr[(j + k) % n]

        if val_i_k == val_j_k:
            k += 1
        else:
            if val_i_k > val_j_k:
                # If arr[i+k] > arr[j+k], then any shift starting at i is larger than shift starting at j.
                # So, i and all shifts between i and i+k are not minimal. Advance i.
                i += k + 1
            else: # val_i_k < val_j_k
                # If arr[i+k] < val_j_k, then any shift starting at j is larger than shift starting at i.
                # So, j and all shifts between j and j+k are not minimal. Advance j.
                j += k + 1
            
            # Make sure i and j are distinct. If they become equal, advance j.
            if i == j:
                j += 1
            k = 0 # Reset matched length

    return min(i, j)

def get_k_th_perm_idx(start_idx, k_exponent, P_powers, max_log_n):
    """
    Computes P_arr^k_exponent[start_idx] using binary lifting.
    """
    curr = start_idx
    for p in range(max_log_n):
        if (k_exponent >> p) & 1:
            curr = P_powers[p][curr]
    return curr

def solve():
    N = int(input())
    P = list(map(int, input().split()))
    A_0 = list(map(int, input().split()))

    # Convert P to 0-indexed P_arr for convenience
    P_arr = [p - 1 for p in P]

    # Step 2: Find disjoint cycles and extract initial values for each cycle
    visited = [False] * N
    cycle_data_list = []  # Stores (list of initial values in cycle, cycle length)

    for i in range(N):
        if not visited[i]:
            current_cycle_indices = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                current_cycle_indices.append(curr)
                curr = P_arr[curr]

            cycle_len = len(current_cycle_indices)
            values_in_cycle = [A_0[idx] for idx in current_cycle_indices]
            
            cycle_data_list.append((values_in_cycle, cycle_len))

    # Step 3: Calculate (s_x, m_x) for each cycle
    # s_x is the required k mod m_x for the specific cycle to be lexicographically smallest
    congruences = [] # List of (s_x, m_x) pairs
    
    for values_in_cycle, cycle_len in cycle_data_list:
        if cycle_len == 0: continue # Should not happen based on constraints N >= 1
        
        s_x = smallest_cyclic_shift_index(values_in_cycle)
        congruences.append((s_x, cycle_len))

    # Step 4: Solve CRT for k_opt
    # We are solving a system k = s_x (mod m_x) for all cycles.
    # A global k must exist given the problem statement (a single number of operations).
    k_opt = 0
    current_lcm = 1

    for s_x, m_x in congruences:
        # We need to find y such that (k_opt + y * current_lcm) = s_x (mod m_x)
        # Rearranging: y * current_lcm = (s_x - k_opt) (mod m_x)
        
        # Calculate target_rem, ensuring it's non-negative
        target_rem = (s_x - k_opt) % m_x
        if target_rem < 0:
            target_rem += m_x
        
        # Use Extended Euclidean Algorithm to solve the linear congruence
        g, inv_current_lcm_candidate, _ = extended_gcd(current_lcm, m_x)
        
        # We assume consistency (target_rem % g == 0) as a solution is guaranteed to exist.
        
        # Calculate y_0, the specific solution for y
        # inv_current_lcm_candidate can be negative, so we adjust it to be positive within its modulo.
        inv_val_positive = (inv_current_lcm_candidate % (m_x // g) + (m_x // g)) % (m_x // g)
        
        y_0 = (target_rem // g * inv_val_positive) % (m_x // g)
        
        # Update k_opt based on the solution y_0
        k_opt += y_0 * current_lcm
        
        # Update current_lcm to LCM(old_lcm, m_x)
        current_lcm = current_lcm * (m_x // g) 
    
    # Ensure k_opt is the smallest non-negative integer solution
    k_opt %= current_lcm 

    # Step 5: Construct A_final using binary lifting
    # Calculate MAX_LOG_N for binary lifting: number of bits needed for N-1
    MAX_LOG_N = (N - 1).bit_length() 
    if N == 1: # (N-1).bit_length() is 0 for N=1. But k_opt can be 0. Loop range(0) is fine.
        MAX_LOG_N = 1 # ensure P_powers has at least P_powers[0]

    P_powers = [[0] * N for _ in range(MAX_LOG_N)]
    P_powers[0] = P_arr[:] # Copy P_arr for the base power 2^0

    # Precompute P_arr^(2^p)
    for p in range(1, MAX_LOG_N):
        for i in range(N):
            P_powers[p][i] = P_powers[p-1][P_powers[p-1][i]]

    A_final = [0] * N
    for j in range(N):
        # A_k[j] = A_0[P_arr^k[j]]
        target_original_index_in_A0 = get_k_th_perm_idx(j, k_opt, P_powers, MAX_LOG_N)
        A_final[j] = A_0[target_original_index_in_A0]

    # Step 6: Print A_final
    sys.stdout.write(" ".join(map(str, A_final)) + "
")

solve()