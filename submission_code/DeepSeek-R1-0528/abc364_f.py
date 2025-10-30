import sys
from heapq import heappush, heappop
from collections import defaultdict

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    q = int(data[1])
    queries = []
    index = 2
    L_arr = []
    R_arr = []
    C_arr = []
    for i in range(q):
        l = int(data[index])
        r = int(data[index+1])
        c = int(data[index+2])
        index += 3
        queries.append((l, r, c))
        L_arr.append(l)
        R_arr.append(r)
        C_arr.append(c)
    
    add_events = [[] for _ in range(n+2)]
    rem_events = [[] for _ in range(n+2)]
    for i in range(q):
        l, r, c = queries[i]
        add_events[l].append(c)
        if r+1 <= n:
            rem_events[r+1].append(c)
    
    base_cost = 0
    active = []
    removal_count = defaultdict(int)
    min_cost_list = [10**18] * (n+1)
    covered = True
    for j in range(1, n+1):
        for c_val in rem_events[j]:
            removal_count[c_val] += 1
        for c_val in add_events[j]:
            heappush(active, c_val)
        while active and removal_count.get(active[0], 0) > 0:
            top = active[0]
            heappop(active)
            removal_count[top] -= 1
            if removal_count[top] == 0:
                del removal_count[top]
        if not active:
            covered = False
            break
        min_cost_j = active[0]
        min_cost_list[j] = min_cost_j
        base_cost += min_cost_j
    
    if not covered:
        print(-1)
        return
        
    total_aux = sum(C_arr)
    min_aux = min(C_arr)
    additional_cost = total_aux - min_aux
    total_cost = base_cost + additional_cost
    print(total_cost)

if __name__ == "__main__":
    main()