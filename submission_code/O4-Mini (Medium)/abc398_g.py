import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    g = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v = map(int, input().split())
        g[u].append(v)
        g[v].append(u)

    color = [-1] * (N+1)
    visited = [False] * (N+1)

    comps = []  # list of (a_i, b_i)
    from collections import deque
    for i in range(1, N+1):
        if not visited[i]:
            # BFS this component
            dq = deque()
            dq.append(i)
            visited[i] = True
            color[i] = 0
            cnt0 = 1
            cnt1 = 0
            while dq:
                u = dq.popleft()
                for w in g[u]:
                    if not visited[w]:
                        visited[w] = True
                        color[w] = color[u] ^ 1
                        if color[w] == 0:
                            cnt0 += 1
                        else:
                            cnt1 += 1
                        dq.append(w)
                    else:
                        # graph guaranteed bipartite, no need to check conflict
                        pass
            comps.append((cnt0, cnt1))

    # M parity
    m_par = M & 1
    n_par = N & 1

    # If N is odd:
    #   total_moves parity = (0) XOR m_par  => equals m_par
    #   so first player wins iff m_par == 1
    if n_par == 1:
        if m_par == 1:
            print("Aoki")
        else:
            print("Takahashi")
        return

    # N even:
    # Check if any component is "flexible" => a_i %2 != b_i %2
    has_flex = False
    # Also compute S_fixed_parity when delta==0
    S_fixed_parity = 0
    for a, b in comps:
        if ((a ^ b) & 1) == 1:
            has_flex = True
        else:
            # both a and b same parity, contribution to S mod2 is fixed = a mod2
            S_fixed_parity ^= (a & 1)

    # If there is a flexible component, first player can force S parity to suit
    if has_flex:
        # Aoki can force a win
        print("Aoki")
    else:
        # S parity is fixed = S_fixed_parity
        # total_moves_parity = (S odd) XOR m_par
        total_par = S_fixed_parity ^ m_par
        if total_par == 1:
            print("Aoki")
        else:
            print("Takahashi")


if __name__ == "__main__":
    main()