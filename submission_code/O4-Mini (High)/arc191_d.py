import sys
import threading

def main():
    import sys
    from array import array
    from collections import deque
    import heapq

    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N = int(line[0]); M = int(line[1]); S = int(line[2]); T = int(line[3])
    S0 = S - 1
    T0 = T - 1

    # Build directed graph with vertex‑splitting:
    # For each original vertex v (0..N-1), we create v_in = v and v_out = v+N.
    # We add a splitting edge v_in->v_out of cost 0.
    # For each undirected edge u–v, add directed edges u_out->v_in and v_out->u_in of cost 1.
    nG = 2 * N
    head = array('i', [-1]) * nG
    to   = array('i')  # to[e] = head of edge e
    cost = array('i')  # cost[e] = weight of edge e (0 or 1)
    nxt  = array('i')  # nxt[e] = next edge index from same source

    # Helper to add a directed edge u->v with cost c
    def add_edge(u, v, c):
        eid = len(to)
        to.append(v)
        cost.append(c)
        nxt.append(head[u])
        head[u] = eid

    # Splitting edges v_in->v_out of cost 0
    for v in range(N):
        add_edge(v, v + N, 0)

    # Original graph edges (both directions)
    for _ in range(M):
        u0, v0 = map(int, data.readline().split())
        u0 -= 1; v0 -= 1
        u_out = u0 + N; v_in = v0
        add_edge(u_out, v_in, 1)
        v_out = v0 + N; u_in = u0
        add_edge(v_out, u_in, 1)

    # 0-1 BFS from s = S_out to compute dist1[]
    INF = 10**9
    s = S0 + N
    t = T0
    dist1 = array('i', [INF]) * nG
    prev_v = array('i', [-1]) * nG  # to reconstruct path
    prev_e = array('i', [-1]) * nG  # edge used to get here

    dist1[s] = 0
    dq = deque([s])
    head_arr = head; to_arr = to; cost_arr = cost; nxt_arr = nxt
    dist1_arr = dist1; prev_v_arr = prev_v; prev_e_arr = prev_e

    while dq:
        u = dq.popleft()
        du = dist1_arr[u]
        e = head_arr[u]
        while e != -1:
            v = to_arr[e]
            nd = du + cost_arr[e]
            if dist1_arr[v] > nd:
                dist1_arr[v] = nd
                prev_v_arr[v] = u
                prev_e_arr[v] = e
                if cost_arr[e] == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
            e = nxt_arr[e]

    # If T_in unreachable, swap impossible
    if dist1_arr[t] == INF:
        print(-1)
        return

    # Reconstruct the shortest path P1 (in the directed graph)
    E = len(to_arr)
    used = bytearray(E)           # mark which edges are on P1
    rev_prev = array('i', [-1]) * nG  # to store reversed‐edge mapping

    vcur = t
    while vcur != s:
        u_prev = prev_v_arr[vcur]
        e_id = prev_e_arr[vcur]
        used[e_id] = 1
        rev_prev[vcur] = u_prev
        vcur = u_prev

    # Run Dijkstra in the Suurballe residual graph to find second disjoint path
    dist2 = array('i', [INF]) * nG
    dist2[s] = 0
    hq = [(0, s)]
    dist2_arr = dist2
    used_arr   = used
    rev_arr    = rev_prev
    dist1_loc  = dist1_arr

    while hq:
        d,u = heapq.heappop(hq)
        if d != dist2_arr[u]:
            continue
        if u == t:
            break
        # reversed edge for P1?
        pu = rev_arr[u]
        if pu != -1 and dist2_arr[pu] > d:
            dist2_arr[pu] = d
            heapq.heappush(hq, (d, pu))
        # forward edges not in P1
        du1 = dist1_loc[u]
        e = head_arr[u]
        while e != -1:
            if not used_arr[e]:
                v = to_arr[e]
                # reduced cost: cost[e] + dist1[u] - dist1[v]
                c2 = cost_arr[e] + du1 - dist1_loc[v]
                nd = d + c2
                if dist2_arr[v] > nd:
                    dist2_arr[v] = nd
                    heapq.heappush(hq, (nd, v))
            e = nxt_arr[e]

    if dist2_arr[t] == INF:
        print(-1)
    else:
        # Let L1 = dist1[t], and we found sum(cost2) = dist2[t] = L2 - L1
        # so the total cycle length (L1 + L2) = 2*L1 + (L2 - L1) = 2*dist1[t] + dist2[t]
        ans = 2 * dist1_arr[t] + dist2_arr[t]
        print(ans)

if __name__ == "__main__":
    main()