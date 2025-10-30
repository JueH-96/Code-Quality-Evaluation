def main():
    import sys
    import numpy as np
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    M = N - 1  # number of bus legs
    legs = [ (int(next(it)), int(next(it))) for _ in range(M) ]
    Q = int(next(it))
    qs = [int(next(it)) for _ in range(Q)]
    
    # We wish to “compress” the entire journey of bus rides into a function 
    # F(t) = t + H(t mod L) so that if Takahashi arrives at bus stop 1 at time t,
    # then after catching all buses he will get to bus stop N at time F(t).
    # Then he takes Y time to walk from bus stop N to Aoki's house.
    # Also, note that he has to walk from his house to bus stop 1 in X time.
    # Thus, the answer for a query q is:
    #   answer = (q + X) + H((q + X) mod L) + Y
    #
    # Each bus leg i has the following rule. Suppose the current time is cur.
    # At bus stop i, a bus departs whenever the time is a multiple of P_i.
    # If cur is not such a time, he must wait until the next multiple.
    # Then taking the bus takes T_i additional time.
    # In formula, if f_i is the function for one leg:
    #   f_i(cur) = cur + ((P_i - (cur mod P_i)) mod P_i) + T_i.
    #
    # We wish to compose these functions for all legs.
    # Because each departure “waiting” depends only on the remainder mod P_i,
    # and each P_i is at most 8, a safe period is the lcm(1,2,...,8)=840.
    # That is, the total extra time appended after processing all legs depends on (cur mod 840),
    # and can be precomputed for each r in 0..839.
    #
    # Let H[r] denote the total extra time (waiting plus ride times) added when entering the first bus leg
    # with time having residue r modulo 840.
    # We compute H by initializing H[r] = 0 and then processing each leg:
    #  new_H[r] = H[r] + ((P - ((r + H[r]) mod P)) mod P) + T.
    #
    # We use numpy to vectorize the update for speed.

    L = 840  # Period; lcm(1,2,...,8)
    # H[r] will be stored for r=0...L-1 and represent cumulative extra time from bus legs.
    H = np.zeros(L, dtype=np.int64)
    r_vals = np.arange(L, dtype=np.int64)
    
    for (P, T) in legs:
        # For a given leg, current time (as focal point for residue r) is (r + H[r]).
        # The waiting time needed is (P - (current mod P)) mod P.
        current = r_vals + H
        wait = (P - (current % P)) % P
        # Update: new cumulative added time for initial residue r:
        H = H + wait + T

    # For each query, the timeline is:
    #   - Leave house at time q.
    #   - Walk to bus stop 1 (X time): start = q + X.
    #   - Take buses: arrival time = start + H[start mod L].
    #   - Walk from bus stop N to Aoki's house (Y time)
    # Thus: answer = (q + X) + H[(q + X) mod L] + Y.
    out_lines = []
    for q in qs:
        start = q + X
        residue = start % L
        answer = start + int(H[residue]) + Y
        out_lines.append(str(answer))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == "__main__":
    main()