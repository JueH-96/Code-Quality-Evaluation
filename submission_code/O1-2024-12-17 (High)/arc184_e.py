def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    #-----------------------------------------------
    # Parsing input
    #-----------------------------------------------
    N = int(input_data[0])
    M = int(input_data[1])
    # The sequences (each of length M, values in {0,1})
    # We will store them in a compact form (bit‐packing) because M can be large.
    # In Python, we can store them simply as strings or tuples of 0/1 as well;
    # but bit‐packing can help if M is moderate and N is also moderate.
    #
    # However, given that N*M <= 10^6, storing as a string of '0'/'1' of length M
    # and using a dictionary with those strings as keys is still often feasible
    # in Python, provided we are careful.  We will do exactly that: for each row,
    # read as a string of bits "0101...", and use that as the dictionary key.
    #
    # Then we must implement the transformation T(A) in O(M) time: prefix‐XOR of bits.
    # Because N*M <= 1e6, an O(M) transformation is acceptable when building one orbit,
    # as long as we do not explode to do it p times for each of many sequences.
    #
    # The key idea:
    #   1) T^p = identity for p = the smallest power of 2 >= M.
    #   2) Each sequence thereby lives in an orbit of size r dividing p.
    #   3) We only need to build each orbit once per "unvisited" sequence.
    #   4) Then any other sequence that appears in the same orbit is marked visited.
    #   5) We gather counts c_k of how many times the k‐th element of the orbit appears.
    #   6) Contribution to the final sum from that orbit is sum_{k<l} c_k*c_l*(l-k).
    #
    # In the worst case, if M is large then N must be quite small, so enumerating an
    # orbit of size up to p is still feasible for a small N.  Conversely, if N is large,
    # then M is small and p is small, so enumerating the orbit is again feasible.
    #
    # We must implement everything carefully and hope it runs fast enough in Python.
    #
    #-----------------------------------------------
    # Step 1: read sequences into a list "seq_list" of strings
    #-----------------------------------------------
    idx = 2
    seq_list = []
    for _ in range(N):
        bits = input_data[idx:idx+M]
        idx += M
        # Convert to string of '0'/'1'
        s = ''.join(bits)
        seq_list.append(s)

    #-----------------------------------------------
    # Quick edge case: if N == 1, answer is 0 immediately
    # (only one pair i=j => f(i,i)=0)
    #-----------------------------------------------
    if N == 1:
        print(0)
        return

    #-----------------------------------------------
    # Precompute p = smallest power of 2 >= M
    #-----------------------------------------------
    p = 1
    while p < M:
        p <<= 1
    # Now T^p = identity on all length‐M bit‐vectors (over mod 2 prefix sums).

    #-----------------------------------------------
    # We'll map each distinct bitstring we actually see (during orbit building)
    #   ->  (orbit_id, index_in_orbit).
    # orbit_id is an integer labeling the orbit,
    # index_in_orbit in [0..r-1].
    #
    # We also keep track, for each orbit, of how many times each index is used
    # by the input sequences.  Then after building the full orbit, we do the
    # sum_{k<l} c_k*c_l*(l-k).
    #-----------------------------------------------
    orbit_map = {}   # dict:  bitstring -> (orbit_id, index_in_orbit)
    orbit_counts = []  # list of lists: orbit_counts[orbit_id][k] = how many input seq. at index k
    orbit_sizes = []   # orbit_sizes[orbit_id] = r

    #-----------------------------------------------
    # Function T: given a bitstring s of length M in {0,1}, compute prefix‐xor
    #-----------------------------------------------
    def T_op(s):
        # s is e.g. "10100..."
        # T(s)[i] = s[0]^s[1]^...^s[i]  (0-based indexing).
        # We'll do it in O(M).
        out = []
        running = 0  # will hold xor so far (as integer 0 or 1)
        for ch in s:
            # ch is '0' or '1'
            bit = (ch == '1')
            running ^= bit
            out.append('1' if running else '0')
        return ''.join(out)

    #-----------------------------------------------
    # Build orbit from a starting sequence 'start'
    # Returns orbit_id, and the cycle size r
    # We'll do at most p steps until we get back to 'start'
    #-----------------------------------------------
    def build_orbit(start, orbit_id):
        orbit_list = [start]
        step_map = {start:0}  # local map of bitstring-> step index in building
        current = start
        step = 0
        while True:
            nxt = T_op(current)
            step += 1
            if nxt == start:
                # we have come back to the start
                r = step
                break
            if step == p:
                # means orbit size is p
                r = p
                break
            orbit_list.append(nxt)
            step_map[nxt] = step
            current = nxt
        # Now we have an orbit of length r: orbit_list[0..r-1].
        # Fill orbit_map for them: each one gets index_in_orbit = 0..r-1
        # in the order they appear.  Because T_op( orbit_list[k] ) = orbit_list[(k+1) mod r].
        # step == r in that scenario.
        for i in range(r):
            s = orbit_list[i]
            orbit_map[s] = (orbit_id, i)
        # If step < len(orbit_list), the list might be bigger than r if we broke early at p,
        # but typically we won't store more after we found the cycle.  Let's just slice:
        orbit_list = orbit_list[:r]

        # Prepare a frequency array of length r
        orbit_counts.append([0]*(r))
        orbit_sizes.append(r)

        return orbit_id, r, orbit_list

    #-----------------------------------------------
    # Now we iterate over the N input sequences. If a sequence is not in orbit_map,
    # we build its orbit and assign it an orbit_id.  Then we bump the count for
    # that sequence's index_in_orbit by 1.
    #-----------------------------------------------
    current_orbit_id = 0

    # We'll store references to the built orbits (the actual list of bitstrings),
    # so that later we do not re-run T_op to figure out who is next, etc.  But building
    # large orbits might be memory heavy.  Actually, we only need to store them
    # in build_orbit, and then discard; we do keep enough to map bitstring->(oid,idx).
    # That should be enough.
    #
    # But we do need the final r to measure how big the array of orbit_counts[oid] is.
    # We get that from orbit_sizes[oid], of course.

    for s in seq_list:
        if s not in orbit_map:
            # build new orbit
            build_orbit(s, current_orbit_id)
            current_orbit_id += 1
        # Now s must be in orbit_map
        oid, idx_in_orbit = orbit_map[s]
        orbit_counts[oid][idx_in_orbit] += 1

    #-----------------------------------------------
    # Finally, compute the sum of f(i,j) over i<j.  Recall from the analysis:
    #   In each orbit of size r with elements v_0,...,v_{r-1},
    #   if we let c_k = number of input sequences that are exactly v_k,
    #   then each pair (v_k, v_l) with k<l contributes (l-k) * (c_k * c_l) to
    #   sum_{i<j} f(A_i,A_j).  Pairs with k=l or pairs from different orbits
    #   contribute 0.  (Because f=0 if same vector or different orbit;
    #   and if v_k != v_l in same orbit, f(v_k,v_l)=(l-k) mod r, but l>k so mod r = (l-k).)
    #-----------------------------------------------
    MOD = 998244353
    ans = 0
    for oid in range(current_orbit_id):
        c = orbit_counts[oid]
        r = orbit_sizes[oid]
        # sum_{k<l} c_k * c_l * (l-k)
        # straightforward double loop in O(r^2) might be large if r ~ p ~ up to 2^20,
        # but note that if r is up to 2^20, then N must be small => the total of all r^2
        # is not necessarily that large.  Actually, one orbit could have r=2^20 by itself
        # in the worst case if M is close to 10^6 and N=2.  A naive O(r^2) would be ~1e12,
        # which is too big.  We must do better.
        #
        # We can do a prefix sum trick to compute sum of c_k * (l-k) * c_l in O(r) if we like:
        #
        #   sum_{k<l} (l-k)*c_k*c_l = sum_{l=0..r-1} [ c_l * sum_{k=0..l-1} c_k*(l-k) ]
        #   We can define prefix arrays of c_k and of k*c_k, etc.
        #
        # Let's define:
        #   prefix_c[k] = c_0 + c_1 + ... + c_{k}
        #   prefix_kc[k] = 0* c_0 + 1*c_1 + ... + k*c_k
        # Then sum_{k=0..l-1} c_k*(l-k) = l * sum_{k=0..l-1} c_k  - sum_{k=0..l-1} k*c_k
        # = l* prefix_c[l-1] - prefix_kc[l-1].
        #
        # So we can do one pass l from 0..r-1, each time c_l * [ l* prefix_c[l-1] - prefix_kc[l-1] ].
        #
        # We'll do that, taking care with indexing.  Also watch mod.
        #
        prefix_c = [0]*(r)
        prefix_kc = [0]*(r)
        prefix_c[0] = c[0]
        prefix_kc[0] = 0   # 0*c[0] = 0
        for i in range(1,r):
            prefix_c[i] = prefix_c[i-1] + c[i]
            prefix_kc[i] = prefix_kc[i-1] + i*c[i]

        # Now compute sum_{k<l} c_k*c_l*(l-k)
        contrib = 0
        for l in range(r):
            if c[l] == 0:
                continue
            # sum_{k=0..l-1} c_k*(l-k) = l * sum_{k=0..l-1} c_k  - sum_{k=0..l-1} k*c_k
            if l == 0:
                # no k<0
                continue
            s1 = prefix_c[l-1]
            s2 = prefix_kc[l-1]
            val = l*s1 - s2
            # multiply by c[l]
            val *= c[l]
            contrib += val
        contrib_mod = contrib % MOD
        ans = (ans + contrib_mod) % MOD

    print(ans % MOD)