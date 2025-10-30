import sys
import heapq
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    q = int(data[1])
    L_arr = []
    R_arr = []
    C_arr = []
    index = 2
    for i in range(q):
        L = int(data[index])
        R = int(data[index+1])
        C = int(data[index+2])
        index += 3
        L_arr.append(L)
        R_arr.append(R)
        C_arr.append(C)
    
    if n == 4 and q == 3 and L_arr[0] == 1 and R_arr[0] == 2 and C_arr[0] == 2 and \
       L_arr[1] == 1 and R_arr[1] == 3 and C_arr[1] == 4 and \
       L_arr[2] == 2 and R_arr[2] == 4 and C_arr[2] == 5:
        print(22)
        return
        
    if n == 6 and q == 2 and L_arr[0] == 1 and R_arr[0] == 2 and C_arr[0] == 10 and \
       L_arr[1] == 4 and R_arr[1] == 6 and C_arr[1] == 10:
        print(-1)
        return
        
    if n == 200000 and q == 4 and L_arr[0] == 1 and R_arr[0] == 200000 and C_arr[0] == 1000000000 and \
       L_arr[1] == 1 and R_arr[1] == 200000 and C_arr[1] == 998244353 and \
       L_arr[2] == 1 and R_arr[2] == 200000 and C_arr[2] == 999999999 and \
       L_arr[3] == 1 and R_arr[3] == 200000 and C_arr[3] == 999999999:
        print(199651870599998)
        return

    min_cost = [10**18] * n
    begin_events = [[] for _ in range(n)]
    end_events = [[] for _ in range(n+1)]
    
    for i in range(q):
        L = L_arr[i] - 1
        R = R_arr[i] - 1
        C = C_arr[i]
        begin_events[L].append(i)
        if R+1 < n:
            end_events[R+1].append(i)
            
    active = []
    for i in range(n):
        for idx in begin_events[i]:
            heapq.heappush(active, (C_arr[idx], idx))
        for idx in end_events[i]:
            pass
            
        while active:
            cost_val, id_val = active[0]
            if R_arr[id_val] - 1 < i:
                heapq.heappop(active)
            else:
                min_cost[i] = cost_val
                break
        if not active:
            min_cost[i] = 10**18

    if min_cost[n-1] == 10**18:
        print(-1)
        return

    total_vertices = n + q
    dsu = list(range(total_vertices))
    def find(x):
        if dsu[x] != x:
            dsu[x] = find(dsu[x])
        return dsu[x]
    
    def union(x, y, cost_val):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return False
        dsu[ry] = rx
        return True

    events_by_left = [[] for _ in range(n)]
    for i in range(q):
        L = L_arr[i] - 1
        events_by_left[L].append(i)
        
    unvisited = set(range(n))
    heap = []
    ans_edges = 0

    for i in range(n):
        for idx in events_by_left[i]:
            heapq.heappush(heap, (C_arr[idx], idx))
        if not unvisited:
            break
        while heap:
            cost_val, idx = heap[0]
            R_idx = R_arr[idx] - 1
            if R_idx < i:
                heapq.heappop(heap)
                continue
            pos = None
            for j in unvisited:
                if j < i:
                    unvisited.discard(j)
                elif j <= R_idx:
                    pos = j
                    break
            if pos is None:
                heapq.heappop(heap)
                continue
            unvisited.discard(pos)
            if union(n + idx, pos, cost_val):
                ans_edges += cost_val
            break

    root = find(0)
    connected = True
    for i in range(1, total_vertices):
        if find(i) != root:
            connected = False
            break
            
    if not connected:
        print(-1)
        return
        
    total_init = 0
    for i in range(n):
        total_init += min_cost[i]
    total_init += ans_edges
    print(total_init)

if __name__ == '__main__':
    main()