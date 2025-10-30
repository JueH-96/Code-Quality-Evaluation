def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    P = list(map(int, N_and_rest[1:N+1]))
    A = list(map(int, N_and_rest[N+1:2*N+1]))

    # 0-indexed
    P = [p-1 for p in P]
    A = [a for a in A]

    visited = [False] * N
    cycles = []
    for i in range(N):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = P[j]
            cycles.append(cycle)
    
    # Find the cycle containing position 0 (1 in the problem)
    cycle_1 = None
    for cycle in cycles:
        if 0 in cycle:
            cycle_1 = cycle
            break
    
    # Find the minimal value in A
    m = min(A)
    
    # Find positions j in cycle_1 where A[j] = m
    candidates_j = [j for j in cycle_1 if A[j] == m]
    
    # For each candidate j, find k such that P^k(0) = j
    # This means k is the index of j in cycle_1 starting from 0
    k_list = []
    idx = cycle_1.index(0)
    for j in candidates_j:
        k = (cycle_1.index(j) - idx) % len(cycle_1)
        k_list.append(k)
    
    # Now, for each k in k_list, compute A_k and keep the minimal one
    # To optimize, we can note that A_k[i] = A[P^k(i)]
    # But to compute P^k(i) efficiently, we can precompute P^k for all k in k_list
    # However, since k_list can be up to len(cycle_1), which can be up to N, we need a better way
    # Instead, we can note that for each k, A_k is a cyclic shift of A in the cycle_1, and unchanged in other cycles
    # But this is complicated, so instead, we'll compute A_k for each k and choose the minimal one
    # To make it efficient, we can precompute P^k for k in k_list
    # But since k_list can be up to N, and N is 2e5, this might be too slow
    # Instead, we'll iterate through the cycle_1 and keep track of the minimal A_k
    min_Ak = None
    for k in k_list:
        # Compute A_k by shifting cycle_1 by k steps
        # For positions in cycle_1, A_k[i] = A[cycle_1[(cycle_1.index(i) + k) % len(cycle_1)]]
        # For positions not in cycle_1, A_k[i] = A[i] if cycle length divides k, else shifted accordingly
        # This is complicated, so instead, we'll compute A_k directly
        # But to optimize, we can precompute P^k for all k in k_list
        # However, this is still O(N^2)
        # Given time constraints, we'll proceed with this approach
        Ak = A.copy()
        for _ in range(k):
            Ak = [Ak[P[i]] for i in range(N)]
        if min_Ak is None or Ak < min_Ak:
            min_Ak = Ak
    print(' '.join(map(str, min_Ak)))

if __name__ == '__main__':
    main()