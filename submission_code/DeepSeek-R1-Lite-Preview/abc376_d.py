from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    adj = [[] for _ in range(N + 1)]
    predecessors = []
    index = 2
    for _ in range(M):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        adj[a].append(b)
        if b == 1:
            predecessors.append(a)
    
    # BFS from vertex 1
    distance = [-1] * (N + 1)
    distance[1] = 0
    queue = deque([1])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if distance[v] == -1:
                distance[v] = distance[u] + 1
                queue.append(v)
    
    # Find the minimum cycle length
    min_cycle = float('inf')
    for v in predecessors:
        if distance[v] != -1:
            cycle_length = distance[v] + 1
            if cycle_length < min_cycle:
                min_cycle = cycle_length
    
    if min_cycle != float('inf'):
        print(min_cycle)
    else:
        print(-1)

if __name__ == "__main__":
    main()