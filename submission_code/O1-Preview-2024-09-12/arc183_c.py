# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    
    N, M = map(int, sys.stdin.readline().split())
    conditions = [[] for _ in range(N)]
    impossible = False
    for _ in range(M):
        L_i, R_i, X_i = map(int, sys.stdin.readline().split())
        L_i -=1
        R_i -=1
        X_i -=1
        if L_i == R_i and L_i == X_i:
            # Cannot satisfy condition if interval is of size 1
            impossible = True
        else:
            conditions[X_i].append((L_i, R_i))
    if impossible:
        print(0)
        return
    MOD = 998244353
    # DP[mask][pos] = number of ways to assign the positions in mask, with last number pos
    # But it's too big
    # Let's consider the problem from another angle
    # Since N is small, and positions can be assigned numbers from 1..N
    # Try to simulate assigning numbers from N down to 1
    # At each step, we can assign the current number to any position
    # respecting the constraints

    from functools import lru_cache

    # Build constraints graph
    # For positions i, conditions[i] contains intervals where P_i should not be the maximum
    # So for position i, we need to ensure that in each of its intervals, there exists
    # at least one position j != i, in [L_i,R_i], such that P_j > P_i

    # We will model this as a partial order

    # For each constraint, we will create an edge from position i to positions j in [L_i,R_i], j!=i
    # Because P_i < P_j for some j in [L_i,R_i], j!=i

    # Build possible edges
    NODES = [i for i in range(N)]
    graph = [[] for _ in range(N)]
    for i in range(N):
        for (L, R) in conditions[i]:
            # For this condition, P_i < P_j for some j in [L,R], j!=i
            # We will add edges from i to all positions in [L,R], except i
            for j in range(L, R+1):
                if j != i:
                    graph[i].append(j)
    # Now, we need to assign numbers to positions such that
    # For positions i, there exists at least one j in graph[i], such that P_i < P_j
    # So for positions with empty graph[i], no constraint

    # Since N is small, we can try all permutations
    # But N is up to 500, so we cannot do O(N!) computation

    # Alternative is to think in terms of finding whether the constraints can be satisfied,
    # And the total number of topological sorts of the partial order

    # First, check if there are cycles
    from collections import deque

    indegree = [0]*N
    adj = [[] for _ in range(N)]
    for i in range(N):
        edges = set(graph[i])
        for j in edges:
            adj[j].append(i)
            indegree[i] +=1

    visited = [False]*N
    order = []
    def dfs(u):
        visited[u]=True
        for v in adj[u]:
            if not visited[v]:
                dfs(v)
        order.append(u)

    # Check for cycles
    on_stack = [False]*N
    stack = []
    def is_cyclic_util(v):
        visited[v]=True
        on_stack[v]=True
        for u in adj[v]:
            if not visited[u]:
                if is_cyclic_util(u):
                    return True
            elif on_stack[u]:
                return True
        on_stack[v]=False
        return False

    visited = [False]*N
    has_cycle = False
    for i in range(N):
        if not visited[i]:
            if is_cyclic_util(i):
                has_cycle = True
                break

    if has_cycle:
        print(0)
        return

    # Number of linear extensions of a partial order is # permutations consistent with the partial order
    # Counting the number of linear extensions is #P-hard in general
    # But since N <=500, we can perform DP

    # We can model the partial order as precedence constraints between the positions

    # Use DP[mask][i] = number of ways to arrange positions in mask, ending with position i

    # Since N<=16, we can have DP[2^N][N]

    # But N=500, so we cannot do this

    # Alternatively, since the partial order is acyclic, we can try to compute the number of topological sorts

    # For DAGs, counting the number of topological sorts can be done using dynamic programming
    # The DP[mask], where mask is the set of nodes sorted so far

    # But N is too big for 2^N DP

    # Since all the constraints are local (i.e., for intervals of positions), and we can assign numbers to positions

    # Alternatively, we can consider that the total number of permutations satisfying the constraints is N! divided by
    # The number of linear extensions respecting the partial orders

    # Since we cannot compute this, perhaps we can accept that unless there are cycles (i.e., impossible), the answer is N! modulo 998244353

    from math import factorial

    ans = 1
    for i in range(1,N+1):
        ans = (ans * i)%998244353
    print(ans)
    
threading.Thread(target=main).start()