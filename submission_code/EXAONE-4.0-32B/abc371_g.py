import math

def mod_inverse(a, m):
    old_r, r = a, m
    old_s, s = 1, 0
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
    if old_r != 1:
        return None
    return old_s % m

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    P = list(map(int, data[1:1+n]))
    A = list(map(int, data[1+n:1+2*n]))
    
    P = [x-1 for x in P]
    
    vis = [False] * n
    cycles = []
    for i in range(n):
        if not vis[i]:
            cycle = []
            cur = i
            while not vis[cur]:
                vis[cur] = True
                cycle.append(cur)
                cur = P[cur]
            cycles.append(cycle)
            
    cycle_id = [-1] * n
    pos_in_cycle = [-1] * n
    cycle_len = [-1] * n
    for idx, cycle in enumerate(cycles):
        L = len(cycle)
        for j, node in enumerate(cycle):
            cycle_id[node] = idx
            pos_in_cycle[node] = j
            cycle_len[node] = L
            
    M = 1
    rr = 0
    ans = [0] * n
    
    for i in range(n):
        L = cycle_len[i]
        pos = pos_in_cycle[i]
        g = math.gcd(M, L)
        base = (pos + rr) % L
        
        min_val = 10**18
        best_d = -1
        for t in range(0, L // g):
            d = (base + t * g) % L
            node = cycles[cycle_id[i]][d]
            if A[node] < min_val:
                min_val = A[node]
                best_d = d
                
        ans[i] = min_val
        
        D = best_d - pos - rr
        D_mod = D % L
        if D_mod < 0:
            D_mod += L
            
        if D_mod % g != 0:
            D_mod = (D_mod % g + g) % g
            if D_mod % g != 0:
                pass
            else:
                D_mod = (D_mod // g) * g
        D_mod //= g
        Mg = M // g
        Lg = L // g
        
        if Lg == 1:
            t0 = 0
        else:
            inv_mg = mod_inverse(Mg, Lg)
            if inv_mg is None:
                t0 = 0
            else:
                t0 = (D_mod * inv_mg) % Lg
                
        new_rr = rr + t0 * M
        new_M = M * Lg
        new_rr %= new_M
        if new_rr < 0:
            new_rr += new_M
            
        M = new_M
        rr = new_rr
        
    print(" ".join(map(str, ans)))
    
if __name__ == '__main__':
    main()