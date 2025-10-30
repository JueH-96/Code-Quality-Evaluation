def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse the first line
    N = int(input_data[0])
    X = int(input_data[1])
    Y = int(input_data[2])
    
    # Read p_i, t_i
    idx = 3
    p = []
    t = []
    for _ in range(N-1):
        pi = int(input_data[idx]); ti = int(input_data[idx+1])
        p.append(pi)
        t.append(ti)
        idx += 2
    
    # Number of queries
    Q = int(input_data[idx]); idx+=1
    queries = list(map(int, input_data[idx:idx+Q]))
    
    # We will build the "composed" function G = G_{1} ∘ G_{2} ∘ ... ∘ G_{N-1},
    # where each G_i(time) = time + wait_i(time) + T_i, with wait_i(time)
    # = (p_i - (time mod p_i)) mod p_i.  Ultimately, for a departure time T at
    # bus stop 1, the arrival time at bus stop N is G(T).  Then from bus stop N
    # to Aoki's house we add Y.  So overall F_1(T) = G(T) + Y.
    #
    # For fast per-query computation, observe that G(T) differs from T by some
    # offset that depends only on (T mod 840).  (Because all p_i ≤ 8, the final
    # composition is periodic with period at most lcm(1..8)=840.)
    #
    # We keep two arrays of length 840 throughout the composition:
    #   offset[r] = how much G so far adds to an input t≡r (mod 840).
    #   nextpos[r] = G(t) mod 840 if t≡r (mod 840).
    #
    # Initially G is the identity (no change yet), so offset[r]=0, nextpos[r]=r.
    # Then for each i=1..N-1, we compose with G_i.  If at some intermediate step
    # we have H(t)=t+offsetH[r] and H(t) mod 840=nextH[r], then
    #   G_{i}(x) = x + wait_i(x) + T_i.
    # Denote offsetG_i[r] = wait_i(r) + T_i, and newrG_i[r] = (r + offsetG_i[r]) mod 840.
    #
    # The composition (G_i∘H)(t) = G_i( H(t) ) = H(t) + wait_i( H(t) ) + T_i.
    # Let r = t mod 840. Then H(t) = t + offsetH[r]. Also H(t) mod 840 = nextH[r].
    # So (G_i∘H)(t) - t
    #   = offsetH[r] + offsetG_i[ nextH[r] ].
    # and (G_i∘H)(t) mod 840
    #   = newrG_i[ nextH[r] ].
    #
    # Thus newOffset[r] = offsetH[r] + offsetG_i[ nextH[r] ],
    #      newNext[r]   = newrG_i[ nextH[r] ].
    #
    # We do this update in O(840) for each i, so O(840*(N-1)) in total.
    # Then the final composition G(t) = t + finalOffset[t mod 840].
    # Finally, for each departure q, the time that Takahashi arrives is
    # (q + X) + finalOffset[(q + X) mod 840] + Y.
    
    # Prepare arrays for the running composition
    SIZE = 840  # lcm of 1..8
    offset = [0]*SIZE      # offset[r] so far
    nextpos = list(range(SIZE))  # nextpos[r] so far
    
    # Compose transformations one by one
    for i in range(N-1):
        pi = p[i]
        ti = t[i]
        # Prepare G_i's offset and next arrays
        # G_i(t) = t + w_i(t) + T_i, where w_i(t) = (pi - (t mod pi)) mod pi.
        # So for each r in [0..839]:
        # offsetG[r] = w_i(r) + T_i
        # nextG[r]   = (r + offsetG[r]) mod 840
        offsetG = [0]*SIZE
        nextG = [0]*SIZE
        for r in range(SIZE):
            wait = (pi - (r % pi)) % pi
            offsetG[r] = wait + ti
            nextG[r]   = (r + offsetG[r]) % SIZE
        
        # Now compose: newOffset[r] = offset[r] + offsetG[ nextpos[r] ]
        #              newNext[r]   = nextG[ nextpos[r] ]
        newOffset = [0]*SIZE
        newNext = [0]*SIZE
        for r in range(SIZE):
            nxt = nextpos[r]
            newOffset[r] = offset[r] + offsetG[nxt]
            newNext[r]   = nextG[nxt]
        
        offset, nextpos = newOffset, newNext
    
    # Now offset[] describes how much the final composition G adds
    # to t for each residue class r = t mod 840.  So G(t) = t + offset[r] if t mod 840 = r.
    #
    # We also must add the final walk Y to get to Aoki's house.
    # So the final answer for a departure time q from the house is:
    #   arrival = (q + X) + offset[ (q + X) mod 840 ] + Y
    
    out = []
    for q in queries:
        start = q + X
        r = start % SIZE
        ans = start + offset[r] + Y
        out.append(str(ans))
    
    print("
".join(out))