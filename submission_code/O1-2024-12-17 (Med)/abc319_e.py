def main():
    import sys
    data = sys.stdin.read().strip().split()
    idx = 0
    
    N = int(data[idx]); idx += 1
    X = int(data[idx]); idx += 1
    Y = int(data[idx]); idx += 1
    
    M = N - 1  # Number of bus rides
    P = [0]*M
    T = [0]*M
    for i in range(M):
        P[i] = int(data[idx]); idx += 1
        T[i] = int(data[idx]); idx += 1
    
    Q = int(data[idx]); idx += 1
    queries = [0]*Q
    for i in range(Q):
        queries[i] = int(data[idx]); idx += 1
    
    # Because each P_i ≤ 8, the bus schedules repeat in a cycle
    # whose length divides 840 (the LCM of 1..8).  We can exploit
    # the fact that if we shift the start time by 840, all wait
    # times repeat and the final arrival shifts by exactly +840.
    #
    # Hence final_arrival(q) = q + offset[q mod 840], for some
    # precomputed "offset[r]" which encapsulates all waiting+walking.
    
    MOD = 840  # LCM of 1..8
    
    # We'll simulate starting remainder r in [0..839],
    # but we incorporate the initial walk X to bus stop 1 by adding X to r
    # at the very start (mod 840).  We'll keep track of the
    # total bus-wait+travel time in currentCost[r].  Then in the end
    # offset[r] = X + currentCost[r] + Y.  So that if someone starts
    # at absolute time q (q mod 840 = r), final arrival is q + offset[r].
    
    currentRem = [0]*MOD
    currentCost = [0]*MOD
    nextRem = [0]*MOD
    nextCost = [0]*MOD
    
    # Initialize:
    # if we leave the house at "time ≡ r (mod 840)",
    # we arrive at bus stop 1 with remainder (r + X) mod 840, cost so far = 0
    for r in range(MOD):
        currentRem[r] = (r + X) % MOD
        currentCost[r] = 0
    
    # Process each bus ride in order:
    for i in range(M):
        p = P[i]
        t = T[i]
        for r in range(MOD):
            rem = currentRem[r]
            cst = currentCost[r]
            
            # wait time = the smallest nonnegative w s.t. (rem + w) is multiple of p
            # i.e. w = (-rem) mod p, or p - (rem mod p) if rem mod p != 0
            w = rem % p
            if w != 0:
                w = p - w
            
            nc = cst + w + t
            nr = (rem + w + t) % MOD
            
            nextRem[r] = nr
            nextCost[r] = nc
        
        # Swap
        currentRem, nextRem = nextRem, currentRem
        currentCost, nextCost = nextCost, currentCost
    
    # Now, for each r, currentCost[r] = total bus-wait+ride time
    # if we started bus-stop-1 at remainder (r + X) mod 840.
    # The final arrival is q + X + (this bus cost) + Y, minus q from that
    # gives X + currentCost[r] + Y.  That is offset[r].
    
    offset = [0]*MOD
    for r in range(MOD):
        offset[r] = X + currentCost[r] + Y
    
    # Answer queries
    out = []
    for q in queries:
        r = q % MOD
        out.append(str(q + offset[r]))
    
    print('
'.join(out))

# Do not forget to call main()!
if __name__ == "__main__":
    main()