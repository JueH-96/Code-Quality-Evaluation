def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    P = [int(x)-1 for x in data[2:2+N]]  # using zero-indexing
    
    # result permutation array, initially copy of P? Actually we want to compute final permutation after K operations.
    result = [None] * N
    visited = [False]*N

    # find cycles
    for i in range(N):
        if not visited[i]:
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = P[cur]
            cycle_len = len(cycle)
            shift = K % cycle_len
            for j in range(cycle_len):
                result[cycle[j]] = P[cycle[(j+shift)%cycle_len]] if shift != 0 else P[cycle[j]]
                # Actually, careful: we need result[i] = the element after applying operation K times.
                # But note that operation: P[i] becomes P(P_i). Our transformation: newMapping(i) = P^2(i).
                # Let's recalc.
    
    # Wait, reconsider: The operation "for i=1,2,...,N update P_i to P[P_i]" is effectively: new permutation = P composed with P, i.e. squaring.
    # But as described, if we apply operation once, the new permutation Q is Q(i) = P(P(i)).
    # Then after K operations, it means Q = P^(2^K)? But wait, let's re-read the problem:
    # "For i=1,2,...,N, simultaneously update P_i to P_{P_i}" -> That means updating the list using the current permutation.
    # Initially, P = (P_1, ..., P_N). After one operation, 
    # index 1 gets value P_{P_1} but note that positions maintain index ordering.
    # Let A be the array index: then new A[i] = A[A[i]].
    # But if we consider positions from 1 to N, then initial P defines mapping i -> P[i]. And after the update:
    # new P[i] becomes old P[ old P[i] ] = (P composed with itself)(i).
    # So the operation effectively does: P = P ∘ P.
    # If we denote f(i)=P[i], then after first op, f becomes f^2, after 2 op, f becomes f^4, after K op f becomes f^(2^K).
    # However, the provided example does not match f^(2^K) since sample: P=(5,6,3,1,2,4). Then f(1)=5, f(2)=6, f(3)=3, f(4)=1, f(5)=2, f(6)=4.
    # f^2(1)=f(5)=2. f^(2^3)=f^(8)(1) should be computed over cycle length? Let's check cycles.
    # Cycle analysis:{
    #   1->5->2->6->4->1 forms cycle length 5, and 3 is fixed.
    # }
    # Operation says after 3 operations: new P[1] = 6.
    # On cycle (1,5,2,6,4): f^(2^3)(1) = f^8(1)=? 8 mod 5=3, so in cycle: cycle[ index (0+3) mod5] = ? cycle = [1,5,2,6,4]; 1->5->2->6, should yield 6; that matches.
    # So indeed after K operations, mapping is f^(2^K). 
    # So how we combine cycles: For each cycle of length L, the new mapping for element x is: move forward by (2^K mod L) places.
    # So we need to compute shift = 2^K mod L.
    # Modify code accordingly.
    
    # Reinitialize:
    result = [None] * N
    visited = [False]*N
    for i in range(N):
        if not visited[i]:
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = P[cur]
            L = len(cycle)
            # compute step = 2^K mod L
            # since K can be huge, we use pow(2, K, L)
            step = pow(2, K, L)
            for idx, pos in enumerate(cycle):
                # In cycle array order, f^(2^K)(pos) = cycle[(idx+step) % L]
                # And new value at pos is f^(2^K)(pos), but careful: value is position +1?
                result[pos] = cycle[(idx+step) % L]
    # But note: In the permutation array P, elements represent f(i) = value, but we need to output permutation after operation such that
    # new P[i] should equal f^(2^K)(i) + 1 (because we use zero-indexing).
    # Currently, result[i] equals the index (0-indexed) that results.
    # So add 1.
    output = " ".join(str(x+1) for x in result)
    sys.stdout.write(output)

if __name__ == '__main__':
    main()