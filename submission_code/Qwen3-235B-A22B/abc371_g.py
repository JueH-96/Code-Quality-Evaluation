import sys
import math
from math import gcd
from functools import reduce

def readints():
    return list(map(int, sys.stdin.readline().split()))

def lcm(a, b):
    return a * b // gcd(a, b)

def decompose_cycles(perm):
    n = len(perm)
    visited = [False] * n
    cycles = []
    for i in range(n):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = perm[current]
            cycles.append(cycle)
    return cycles

def main():
    N = int(sys.stdin.readline())
    P = list(map(lambda x: int(x)-1, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    
    cycles = decompose_cycles(P)
    
    # For each cycle, collect the elements of A in the cycle's order
    cycle_data = []
    for cycle in cycles:
        l = len(cycle)
        elements = [A[i] for i in cycle]
        cycle_data.append((cycle, elements))
    
    # We need to find the k that minimizes the array after rotating each cycle's elements by k steps
    # Since k must be the same for all cycles, we need to find the best k by comparing all possible k's
    # up to the LCM of all cycle lengths
    # However, this is not feasible for large LCM, so we use a different approach:
    # We process each position in the array in order, and for each position, narrow down the possible k's
    
    # Precompute for each position: its cycle index, offset in the cycle
    pos_info = [None] * N
    for cycle_idx, (cycle, elements) in enumerate(cycle_data):
        for offset, pos in enumerate(cycle):
            pos_info[pos] = (cycle_idx, offset)
    
    # Generate all possible k's up to the maximum possible LCM of cycle lengths
    # But this is not feasible for large N, so we need a smarter way
    # Instead, we process each position in order and narrow down possible k's
    
    # Initialize allowed residues: initially, any k mod 1 (only 0)
    allowed = [(0, 1)]  # list of (residue, modulus)
    
    for pos in range(N):
        # Get cycle info for current position
        cycle_idx, offset = pos_info[pos]
        cycle_elements = cycle_data[cycle_idx][1]
        cycle_len = len(cycle_elements)
        
        # For each allowed (r, m), compute what the value would be for this cycle's rotation k = r + t*m
        # and find the minimal value, then filter allowed residues to those achieving this value
        
        # Build a list of possible (value, residue, modulus) for the new allowed set
        value_to_residues = {}
        
        for (r, m) in allowed:
            # For current allowed residue r mod m, we need to find k = r mod m
            # and k mod cycle_len determines the value
            # We need to find all residues mod new_mod that are compatible with r mod m and give the same cycle rotation
            
            # To compute the rotation in the cycle, we need k mod cycle_len
            # So for all possible k ≡ r (mod m), k mod cycle_len can take various values
            # To find all possible residues mod gcd(m, cycle_len), we can use the following approach:
            
            g = gcd(m, cycle_len)
            mod_lcm = cycle_len // g  # LCM(m, cycle_len) is m * cycle_len // gcd(m, cycle_len)
            
            # The possible residues of k mod cycle_len for k ≡ r mod m are: r mod gcd + s * m mod cycle_len
            residues_in_cycle = set()
            base = r % cycle_len
            step = m
            for s in range(g):
                res = (base + s * step) % cycle_len
                residues_in_cycle.add(res)
            
            # For each possible residue_in_cycle in residues_in_cycle, compute the value
            # and collect the allowed residues that would result in this value
            min_value = min( (cycle_elements[ (offset + k) % cycle_len ] for k in residues_in_cycle), default=None )
            
            for k_mod_cycle in residues_in_cycle:
                value = cycle_elements[ (offset + k_mod_cycle) % cycle_len ]
                if value not in value_to_residues:
                    value_to_residues[value] = []
                
                # We need to find all k that are ≡ r mod m and k ≡ k_mod_cycle mod cycle_len
                # Solve the congruence system:
                # x ≡ r mod m
                # x ≡ k_mod_cycle mod cycle_len
                a1, m1 = r, m
                a2, m2 = k_mod_cycle, cycle_len
                
                # Use Chinese Remainder Theorem for two congruences
                def crt(a1, m1, a2, m2):
                    # Solve for x ≡ a1 mod m1
                    #          x ≡ a2 mod m2
                    g, x, y = extended_gcd(m1, m2)
                    if (a2 - a1) % g != 0:
                        return None  # No solution
                    lcm = m1 // g * m2
                    x = (a1 + (a2 - a1) // g * x * m1) % lcm
                    return (x, lcm)
                
                def extended_gcd(a, b):
                    if b == 0:
                        return (a, 1, 0)
                    else:
                        g, x, y = extended_gcd(b, a % b)
                        return (g, y, x - (a // b) * y)
                
                res = crt(a1, m1, a2, m2)
                if res is not None:
                    new_r, new_mod = res
                    value_to_residues[value].append( (new_r, new_mod) )
        
        # Find the minimal value
        if not value_to_residues:
            break  # No solution, which should not happen
        min_val = min(value_to_residues.keys())
        # Update allowed residues to those that produce min_val
        allowed = value_to_residues[min_val]
        
        # Merge allowed residues with the same residue and modulus to avoid duplication
        # This step is needed to prevent allowed list from growing exponentially
        merged = {}
        for (r, m) in allowed:
            if (r, m) not in merged:
                merged[(r, m)] = True
        allowed = list(merged.keys())
        
        if len(allowed) == 1:
            break  # Only one possibility left
    
    # Now, find the minimal k in allowed residues
    # We can pick the smallest k >=0 that satisfies any of the congruences
    # Since allowed residues may have multiple congruences, we need to find the minimal k
    # For this, we can pick the minimal residue among allowed residues with smallest modulus
    
    # Generate possible k candidates
    candidates = set()
    for (r, m) in allowed:
        # Generate k = r + t*m, find the minimal k >=0
        candidates.add(r % m)
    
    min_k = None
    for k_candidate in candidates:
        # We need to find a k >=0 that is ≡ allowed residues
        # To find the actual minimal k, we can try k_candidate and add multiples of LCM of moduli
        pass
    
    # Brute force approach for small N:
    # Generate all possible k up to LCM of all cycle lengths
    # Compute LCM of all cycle lengths
    cycle_lens = [len(cycle) for cycle, _ in cycle_data]
    total_lcm = reduce(lcm, cycle_lens, 1)
    
    best_array = None
    best_k = 0
    
    # Precompute P^k for all positions for all k up to total_lcm-1
    # Not feasible for large total_lcm, but for the purpose of this code:
    for k in range(total_lcm):
        current = list(A)
        # Apply permutation P k times
        # But we can compute for each position where its value comes from
        arr = [0]*N
        for i in range(N):
            pos = i
            for _ in range(k):
                pos = P[pos]
            arr[i] = A[pos]
        if best_array is None or arr < best_array:
            best_array = arr
            best_k = k
    
    print(' '.join(map(str, best_array)))

if __name__ == '__main__':
    main()