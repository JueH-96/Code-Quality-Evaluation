def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    N, X = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+2*N]))
    P = list(map(int, input_data[2+2*N:2+3*N]))
    Q = list(map(int, input_data[2+3*N:2+4*N]))

    # Convert to 0-based for easier handling
    X -= 1
    P = [p-1 for p in P]
    Q = [q-1 for q in Q]

    #----------------------------------------------------------------
    # 1) Identify the cycle of P that contains X.
    #    We'll store it in cycP (a list of nodes in the order visited),
    #    and also build a dict idxP that maps each node in that cycle to its index in cycP.
    #
    #    Then for each i with A[i] == 1, check if i is in that cycle. If not, impossible.
    #    Otherwise compute the "distance" (number of picks needed) to get from i to X.
    #    Because picking a box moves that color forward by +1 along P (mod cycle length),
    #    the distance from cycP-index(i) to cycP-index(X)=0 is (cycle_length - i_index) mod cycle_length.
    #    We take the maximum over all i with A[i]=1 to get dR.
    #
    # 2) Repeat similarly for Q to get cycQ, idxQ, and dB.
    #
    # 3) If any ball is not in the respective cycle, answer=-1.
    #
    # 4) Otherwise, we compute a possible "synergy" that can reduce the total picks:
    #    - Let CP = set of nodes in cycP, CQ = set of nodes in cycQ.
    #    - If for all u in cycP we have P[u] == Q[u], then red and blue follow exactly the same edges
    #      around that cycle, so synergy = min(dR, dB) (we can move them together every step).
    #    - Else if (CP ∩ CQ has some node != X) and both dR>0 and dB>0, then synergy=1
    #      (we can at least arrange them to coincide in one box and pick it once for both).
    #    - Otherwise synergy=0.
    #
    # 5) The result = dR + dB - synergy.  And of course it must be >= max(dR,dB).
    #
    # 6) Print the result or -1 if impossible.

    sys.setrecursionlimit(10**7)

    # Function to extract the cycle of permutation perm containing start
    # Returns (cycle_list, index_map) where cycle_list is in forward order
    # starting from 'start' and ending just before it repeats, and
    # index_map[node_in_cycle] = index_in_cycle_list.
    def get_cycle(perm, start):
        cycle_list = []
        index_map = {}
        cur = start
        idx = 0
        while True:
            cycle_list.append(cur)
            index_map[cur] = idx
            idx += 1
            cur = perm[cur]
            if cur == start:
                break
        # Make sure start is at index 0. (We constructed it so that happens by design.)
        return cycle_list, index_map

    # Get P-cycle containing X
    cycP, idxP = get_cycle(P, X)
    setP = set(cycP)
    lenP = len(cycP)

    # Compute distance for red (dR)
    dR = 0
    feasible = True
    for i in range(N):
        if A[i] == 1:
            if i not in setP:
                feasible = False
                break
            # distance from cycP-index(i) to cycP-index(X)=0 by going "forward" is:
            k = idxP[i]
            dist = (lenP - k) % lenP  # how many picks to go from i to X
            if dist > dR:
                dR = dist
    if not feasible:
        print(-1)
        return

    # Get Q-cycle containing X
    cycQ, idxQ = get_cycle(Q, X)
    setQ = set(cycQ)
    lenQ = len(cycQ)

    # Compute distance for blue (dB)
    dB = 0
    feasible = True
    for i in range(N):
        if B[i] == 1:
            if i not in setQ:
                feasible = False
                break
            k = idxQ[i]
            dist = (lenQ - k) % lenQ
            if dist > dB:
                dB = dist
    if not feasible:
        print(-1)
        return

    # If there are no balls at all, dR=dB=0 => answer=0 right away
    if dR == 0 and dB == 0:
        print(0)
        return

    # Now compute synergy
    # Check if for all u in cycP, we have P[u] == Q[u]. If cycP and cycQ differ in length,
    # or if they differ on which nodes are in them, that already means not the same cycle.
    # But let's do a direct check among all nodes in cycP: do we have P[u] == Q[u] for each u in cycP?
    # Also must check cycP == cycQ as sets. If they differ, can't be the same cycle structure.
    # If truly identical edge-by-edge along that cycle containing X, synergy = min(dR,dB).
    # Else if intersection (besides X) is non-empty and both dR>0,dB>0 => synergy=1
    # Else synergy=0.
    if lenP == lenQ and setP == setQ:
        # possible that they are the exact same cycle of the same length - check edges
        same_edges = True
        for u in cycP:
            if P[u] != Q[u]:
                same_edges = False
                break
        if same_edges:
            synergy = min(dR, dB)
        else:
            # different edges => partial overlap?
            # if intersection besides X is non-empty and min(dR,dB)>0 => synergy=1
            # else 0
            inter = (setP & setQ) - {X}
            if inter and dR > 0 and dB > 0:
                synergy = 1
            else:
                synergy = 0
    else:
        # lengths differ or sets differ => not the same cycle
        # check intersection
        inter = (setP & setQ) - {X}
        if inter and dR > 0 and dB > 0:
            synergy = 1
        else:
            synergy = 0

    ans = dR + dB - synergy
    # Ensure answer >= max(dR, dB) always
    if ans < max(dR, dB):
        ans = max(dR, dB)

    print(ans)


# Do not forget to call main()!
if __name__ == "__main__":
    main()