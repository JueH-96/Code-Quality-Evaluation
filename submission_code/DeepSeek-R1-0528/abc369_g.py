import sys
import heapq
from collections import deque

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    graph = [[] for _ in range(n+1)]
    index = 1
    for i in range(n-1):
        u = int(data[index]); v = int(data[index+1]); l = int(data[index+2]); 
        index += 3
        graph[u].append((v, l))
        graph[v].append((u, l))
    
    parent_arr = [0] * (n+1)
    dist1 = [0] * (n+1)
    q = deque([1])
    parent_arr[1] = 0
    children_of_1 = []
    
    while q:
        u = q.popleft()
        for (v, l) in graph[u]:
            if v == parent_arr[u]:
                continue
            parent_arr[v] = u
            dist1[v] = dist1[u] + l
            q.append(v)
    
    for i in range(2, n+1):
        if parent_arr[i] == 1:
            children_of_1.append(i)
    
    branches = []
    branches.append([1])
    
    visited = [False] * (n+1)
    visited[1] = True
    for child in children_of_1:
        branch_nodes = []
        q2 = deque([child])
        visited[child] = True
        while q2:
            u = q2.popleft()
            branch_nodes.append(u)
            for (v, l) in graph[u]:
                if v == parent_arr[u] or visited[v]:
                    continue
                visited[v] = True
                q2.append(v)
        branches.append(branch_nodes)
    
    global_activated = [-1] * (n+1)
    global_activated[1] = 1
    
    branch_events = []
    
    for branch in branches:
        nodes_sorted = sorted(branch, key=lambda x: dist1[x], reverse=True)
        events = []
        for node in nodes_sorted:
            x = node
            while x != 1 and global_activated[x] == -1:
                x = parent_arr[x]
            meeting_point = x
            add_val = dist1[node] - dist1[meeting_point]
            events.append(add_val)
            
            cur = node
            while cur != meeting_point:
                global_activated[cur] = cur
                cur = parent_arr[cur]
                
        branch_events.append(events)
    
    heap = []
    for bid, events in enumerate(branch_events):
        if events:
            heapq.heappush(heap, (-events[0], bid, 0))
    
    total_weight = 0
    ans = []
    
    for k in range(1, n+1):
        if not heap:
            ans.append(2 * total_weight)
        else:
            neg_event, bid, event_idx = heapq.heappop(heap)
            event_val = -neg_event
            total_weight += event_val
            ans.append(2 * total_weight)
            if event_idx + 1 < len(branch_events[bid]):
                next_event_val = branch_events[bid][event_idx+1]
                heapq.heappush(heap, (-next_event_val, bid, event_idx+1))
    
    for res in ans:
        print(res)

if __name__ == '__main__':
    main()