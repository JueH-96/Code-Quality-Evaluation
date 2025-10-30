def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N, X = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+2*N]))
    P = list(map(lambda x: int(x)-1, input_data[2+2*N:2+3*N]))  # zero-based
    Q = list(map(lambda x: int(x)-1, input_data[2+3*N:2+4*N]))  # zero-based
    X -= 1  # move X to zero-based

    # Quick check: if there are no balls at all, answer is 0 immediately.
    if all(a == 0 for a in A) and all(b == 0 for b in B):
        print(0)
        return

    # ------------------------------------------------------------
    # We will do the following:
    #
    # 1) For the red balls, we consider the permutation P.
    #    - Find the cycle in P that contains X.
    #    - Any box i with a red ball (A[i] = 1) must lie in the same cycle,
    #      otherwise it's impossible -> print(-1).
    #    - Among those boxes in that cycle, each one (except X itself) must be
    #      picked to move its red ball forward, along the path i -> P[i] -> ...
    #      -> X. In a cycle representation that starts at X (index 0),
    #      if posP[i] = k (k>0), it belongs to the set of picks needed:
    #      i.e. indices from k up to c-1 in that cycle.
    #    - So effectively, for all i with A[i] = 1, we gather the range from
    #      posP[i] up to c-1. The union of these ranges is from
    #      rminP = min(posP[i] for i) (ignoring i == X) up to (c-1).
    #      That set has size (c - rminP) if rminP < c, or 0 if no red ball
    #      except perhaps at X.
    #
    # 2) Do the same for blue balls with Q.
    #
    # 3) Let Rset be the actual set of boxes for red picks, Bset be for blue picks.
    #    The answer is |Rset ∪ Bset|.
    #    Why? Because picking a box once can move all red and blue balls present
    #    in that box at that moment. We can schedule these picks in a way that
    #    each box in Rset ∪ Bset is picked exactly once in some order (outside-in).
    #
    # Complexity: We only need to trace out two cycles (one in P and one in Q),
    # each of length at most N, and build sets. This is efficient for N up to 2e5.
    #
    # Implementation steps:
    #   - Build cycle CP containing X in P, record posP[v] = index in that cycle
    #     (or -1 if not in that cycle).
    #   - Check feasibility for red: every i with A[i] = 1 must have posP[i] != -1,
    #     else -1.
    #   - Find rminP = min(posP[i] for i with A[i]=1 and i!=X). If none, rminP=c.
    #   - Rset = { CP[k] for k in range(rminP, c) }, if rminP < c.
    #   - Similarly for Q and B.
    #   - Answer = size of Rset ∪ Bset. If infeasible, -1.

    # Helper function: given permutation perm and a start x,
    # return the list of boxes in the cycle containing x in the order
    # x -> perm[x] -> perm^2[x] -> ... -> x again,
    # plus a dictionary pos[v] giving index of v in that cycle, or -1 if v not in cycle.
    def get_cycle_and_pos(perm, x):
        # Traverse from x until we loop back to x
        cycle = []
        visited = False
        cur = x
        while True:
            cycle.append(cur)
            cur = perm[cur]
            if cur == x:
                break
        # build pos-dict
        pos_dict = [-1] * N
        for idx, box in enumerate(cycle):
            pos_dict[box] = idx
        return cycle, pos_dict

    # Find cycle CP in P containing X
    cycle_p, posP = get_cycle_and_pos(P, X)
    c = len(cycle_p)

    # Check feasibility for red
    for i in range(N):
        if A[i] == 1:
            if posP[i] < 0:  # i not in cycle with X
                print(-1)
                return

    # Find rminP
    rminP = c  # if it stays c, that means no outside-X red ball
    for i in range(N):
        if A[i] == 1 and i != X:
            rminP = min(rminP, posP[i])
    # Rset = boxes in cycle_p[k] for k in [rminP..c-1] if rminP < c
    Rset = set()
    if rminP < c:
        for k in range(rminP, c):
            Rset.add(cycle_p[k])

    # Find cycle CQ in Q containing X
    cycle_q, posQ = get_cycle_and_pos(Q, X)
    d = len(cycle_q)

    # Check feasibility for blue
    for i in range(N):
        if B[i] == 1:
            if posQ[i] < 0:  # i not in same cycle with X in Q
                print(-1)
                return

    # Find rminQ
    rminQ = d
    for i in range(N):
        if B[i] == 1 and i != X:
            rminQ = min(rminQ, posQ[i])
    # Bset = boxes in cycle_q[k] for k in [rminQ..d-1] if rminQ < d
    Bset = set()
    if rminQ < d:
        for k in range(rminQ, d):
            Bset.add(cycle_q[k])

    # The answer is size of Rset ∪ Bset
    ans = len(Rset.union(Bset))
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()