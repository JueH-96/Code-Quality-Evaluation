from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx + 1])
    idx += 2
    sets = []
    for _ in range(N):
        A = int(data[idx])
        idx += 1
        s = set()
        for _ in range(A):
            s.add(int(data[idx]))
            idx += 1
        sets.append(s)
    
    element_to_sets = [[] for _ in range(M + 1)]
    for set_id, s in enumerate(sets):
        for elem in s:
            if elem <= M:
                element_to_sets[elem].append(set_id)
    
    start_sets = set()
    target_sets = set()
    for set_id, s in enumerate(sets):
        if 1 in s:
            start_sets.add(set_id)
        if M in s:
            target_sets.add(set_id)
    
    if start_sets & target_sets:
        print(0)
        return
    
    visited = [False] * N
    for s in start_sets:
        visited[s] = True
    
    queue = deque()
    for s in start_sets:
        queue.append((s, 0))
    
    while queue:
        current_set, steps = queue.popleft()
        for elem in sets[current_set]:
            if elem <= M:
                for neighbor in element_to_sets[elem]:
                    if not visited[neighbor]:
                        if M in sets[neighbor]:
                            print(steps + 1)
                            return
                        visited[neighbor] = True
                        queue.append((neighbor, steps + 1))
    
    print(-1)

if __name__ == '__main__':
    main()