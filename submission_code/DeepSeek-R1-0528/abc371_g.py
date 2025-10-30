import math
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    A = list(map(int, data[1+n:1+2*n]))
    
    P = [x-1 for x in P]
    
    visited = [False] * n
    cycles = []
    for i in range(n):
        if not visited[i]:
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = P[cur]
            cycles.append(cycle)
    
    cycles_with_min = []
    for cycle in cycles:
        if cycle:
            min_index = min(cycle)
            cycles_with_min.append((min_index, cycle))
    cycles_with_min.sort(key=lambda x: x[0])
    
    res = [0] * n
    current_residue = 0
    current_modulus = 1
    
    for min_index, cycle in cycles_with_min:
        L = len(cycle)
        sorted_indices = sorted(cycle)
        cycle_values = [A[i] for i in cycle]
        pos_in_cycle = {node: idx for idx, node in enumerate(cycle)}
        
        g = math.gcd(current_modulus, L)
        r0 = current_residue % g
        R = []
        t = 0
        r_val = r0
        while r_val < L:
            R.append(r_val)
            t += 1
            r_val = r0 + t * g
        
        if min(cycle_values) == max(cycle_values):
            chosen_r = min(R)
        else:
            candidate_set = R
            for node in sorted_indices:
                if len(candidate_set) == 1:
                    break
                pos = pos_in_cycle[node]
                min_val = 10**18
                next_set = []
                for r in candidate_set:
                    index_in_cycle_vals = (pos + r) % L
                    val = cycle_values[index_in_cycle_vals]
                    if val < min_val:
                        min_val = val
                        next_set = [r]
                    elif val == min_val:
                        next_set.append(r)
                candidate_set = next_set
            chosen_r = min(candidate_set)
        
        g1 = g
        a = current_modulus
        b = chosen_r - current_residue
        L_reduced = L // g1
        b_reduced = b % L
        if b_reduced % g1 != 0:
            b_reduced = (b_reduced % L + L) % L
        if b_reduced % g1 != 0:
            g1_check = math.gcd(a, L)
            if g1 != g1_check:
                g1 = g1_check
                L_reduced = L // g1
                if g1 != 0:
                    b_reduced = b_reduced % g1
                    if b_reduced % g1 != 0:
                        b_reduced = (b_reduced % g1) + g1
        b_reduced = b_reduced // g1
        a_reduced = a // g1
        
        if L_reduced == 0:
            t0 = 0
        else:
            if a_reduced % L_reduced == 0:
                if b_reduced % L_reduced == 0:
                    t0 = 0
                else:
                    t0 = 0
            else:
                try:
                    inv_a = pow(a_reduced, -1, L_reduced)
                except:
                    t0 = 0
                else:
                    t0 = (b_reduced * inv_a) % L_reduced
        x0 = current_residue + t0 * a
        new_modulus = a * L_reduced
        x0 = x0 % new_modulus
        
        current_residue = x0
        current_modulus = new_modulus
        
        for node in cycle:
            pos = pos_in_cycle[node]
            new_index_in_cycle = (pos + chosen_r) % L
            val = cycle_values[new_index_in_cycle]
            res[node] = val
            
    print(" ".join(map(str, res)))
    
if __name__ == "__main__":
    main()