import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        print(-1)
        return
    
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    s = int(next(it))
    t = int(next(it))
    
    graph = [[] for _ in range(n+1)]
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        graph[u].append(v)
        graph[v].append(u)
        
    if s == t:
        print(-1)
        return
        
    forward_queue = deque()
    forward_visited = {}
    backward_queue = deque()
    backward_visited = {}
    
    forward_queue.append((s, t))
    forward_visited[(s, t)] = 0
    backward_queue.append((t, s))
    backward_visited[(t, s)] = 0
    
    if (t, s) in forward_visited:
        print(forward_visited[(t, s)])
        return
        
    ans = None
    
    while forward_queue and backward_queue:
        size = len(forward_queue)
        for _ in range(size):
            a, b = forward_queue.popleft()
            current_step = forward_visited[(a, b)]
            
            for na in graph[a]:
                if na == b:
                    continue
                new_state = (na, b)
                if new_state not in forward_visited:
                    forward_visited[new_state] = current_step + 1
                    forward_queue.append(new_state)
                    if new_state in backward_visited:
                        total_steps = current_step + 1 + backward_visited[new_state]
                        if ans is None or total_steps < ans:
                            ans = total_steps
            for nb in graph[b]:
                if nb == a:
                    continue
                new_state = (a, nb)
                if new_state not in forward_visited:
                    forward_visited[new_state] = current_step + 1
                    forward_queue.append(new_state)
                    if new_state in backward_visited:
                        total_steps = current_step + 1 + backward_visited[new_state]
                        if ans is None or total_steps < ans:
                            ans = total_steps
        if ans is not None:
            print(ans)
            return
            
        size = len(backward_queue)
        for _ in range(size):
            a, b = backward_queue.popleft()
            current_step = backward_visited[(a, b)]
            
            for na in graph[a]:
                if na == b:
                    continue
                new_state = (na, b)
                if new_state not in backward_visited:
                    backward_visited[new_state] = current_step + 1
                    backward_queue.append(new_state)
                    if new_state in forward_visited:
                        total_steps = current_step + 1 + forward_visited[new_state]
                        if ans is None or total_steps < ans:
                            ans = total_steps
            for nb in graph[b]:
                if nb == a:
                    continue
                new_state = (a, nb)
                if new_state not in backward_visited:
                    backward_visited[new_state] = current_step + 1
                    backward_queue.append(new_state)
                    if new_state in forward_visited:
                        total_steps = current_step + 1 + forward_visited[new_state]
                        if ans is None or total_steps < ans:
                            ans = total_steps
        if ans is not None:
            print(ans)
            return
            
    print(-1)

if __name__ == "__main__":
    main()