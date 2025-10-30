def extended_euclid(a, b):
    # Returns (g, x, y) such that a*x + b*y = g = gcd(a, b)
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = extended_euclid(b, a % b)
    return (g, y1, x1 - (a // b) * y1)
    
def crt(a, m, b, n):
    # Combine two congruences: x ≡ a (mod m) and x ≡ b (mod n)
    # Returns (x, mod) where mod = LCM(m, n) and x is the unique solution mod LCM.
    # (Assumes that the system is solvable.)
    from math import gcd
    d = gcd(m, n)
    if (a - b) % d != 0:
        return (-1, -1)  # no solution; should not happen in our usage.
    lcm = m // d * n
    # Solve for t in: a + m*t = b (mod n)
    # That is, find t satisfying m*t ≡ (b - a) (mod n).
    # Write m' = m/d and n' = n/d.
    m_prime = m // d
    n_prime = n // d
    diff = (b - a) % n  # non-negative remainder
    diff //= d
    g, inv, _ = extended_euclid(m_prime, n_prime)
    inv %= n_prime
    t = (diff * inv) % n_prime
    x = a + m * t
    x %= lcm
    return (x, lcm)

def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    P = [int(next(it)) - 1 for _ in range(N)]
    A = [int(next(it)) for _ in range(N)]
    
    # Compute cycles of P.
    visited = [False] * N
    cycles = []  # list of cycles; each cycle is a list of indices in the order given by P.
    # Also record for each index, which cycle it belongs to and its position in that cycle.
    cycle_id = [-1] * N
    pos_in_cycle = [-1] * N
    
    for i in range(N):
        if not visited[i]:
            cur = i
            cyc = []
            while not visited[cur]:
                visited[cur] = True
                cyc.append(cur)
                cur = P[cur]
            # rotate cyc so that the smallest index appears first,
            # since in the final array the first occurrence (i.e. the smallest index) in this cycle determines its lex contribution.
            idx_min = 0
            for j in range(1, len(cyc)):
                if cyc[j] < cyc[idx_min]:
                    idx_min = j
            new_cyc = [ cyc[(idx_min + j) % len(cyc)] for j in range(len(cyc)) ]
            cid = len(cycles)
            cycles.append(new_cyc)
            for j, idx in enumerate(new_cyc):
                cycle_id[idx] = cid
                pos_in_cycle[idx] = j
                
    # Process cycles in increasing order of their minimal index (the anchor).
    anchors = []  # (anchor_index, cycle_id)
    for cid, cyc in enumerate(cycles):
        anchors.append( (cyc[0], cid) )  # by our rotation, the first element is the smallest index in the cycle.
    anchors.sort()
    
    # We now maintain a global constraint on k: at each stage we have k ≡ cur_rem (mod cur_mod).
    cur_mod = 1
    cur_rem = 0  # initially, k is free (mod 1).
    # For each cycle, we want to “force” the rotation of that cycle (that is, k mod (length of cycle))
    # so that the value at its anchor, which is A[ cycle[r] ], is as small as possible.
    # But k must satisfy the current global CRT condition. In other words,
    # if this cycle has length L then our chosen residue r (with 0 <= r < L) must satisfy:
    #      r ≡ cur_rem (mod d)
    # where d = gcd(cur_mod, L).
    # Then among the allowed r’s we choose one which minimizes A[ cycle[r] ].
    from math import gcd
    for anchor, cid in anchors:
        cyc = cycles[cid]
        L = len(cyc)
        d = gcd(cur_mod, L)
        # Allowed residues: all r in 0..L-1 such that r mod d == (cur_rem mod d)
        allowed = []
        best_r = None
        best_val = None
        rem_needed = cur_rem % d
        for r in range(L):
            if r % d == rem_needed:
                # The value at the anchor of this cycle (the smallest index in the cycle) after rotation r is:
                # A[ cyc[r] ]
                val = A[cyc[r]]
                if best_r is None or val < best_val:
                    best_r = r
                    best_val = val
        # Update the global CRT condition: we need k ≡ best_r (mod L) as well as the old condition.
        newRem, newMod = crt(cur_rem, cur_mod, best_r, L)
        cur_rem, cur_mod = newRem, newMod

    # The final chosen operation count is k = cur_rem (the minimum nonnegative solution).
    k = cur_rem

    # Now compute the final array f = A[P^k]:
    # For every cycle, if the cycle has length L then for any index in this cycle that sits at position j in our
    # ordering (cyc list), the final value after k operations is A[ cyc[(j + (k mod L)) mod L] ].
    result = [0] * N
    for cyc in cycles:
        L = len(cyc)
        shift = k % L
        for j in range(L):
            # the index cyc[j] gets the value from cyc[(j + shift) mod L]
            result[ cyc[j] ] = A[ cyc[(j + shift) % L] ]
    # print the result in one line with spaces.
    sys.stdout.write(" ".join(map(str, result)))
 
if __name__ == '__main__':
    main()