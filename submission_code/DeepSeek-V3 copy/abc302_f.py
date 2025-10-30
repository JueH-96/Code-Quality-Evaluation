from collections import defaultdict, deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    sets = []
    for _ in range(N):
        A = int(data[idx])
        idx += 1
        S = list(map(int, data[idx:idx+A]))
        idx += A
        sets.append(S)
    
    # Create a graph where nodes are sets, and edges connect sets with common elements
    # But to optimize, we can create a graph where nodes are elements, and edges connect elements that are in the same set
    # Then, we need to find the shortest path from 1 to M in this graph
    
    # Create adjacency list for elements
    adj = defaultdict(set)
    for s in sets:
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                adj[s[i]].add(s[j])
                adj[s[j]].add(s[i])
    
    # BFS to find the shortest path from 1 to M
    visited = {}
    queue = deque()
    queue.append(1)
    visited[1] = 0
    
    while queue:
        current = queue.popleft()
        if current == M:
            # The number of operations is the number of steps minus one
            # Because each step corresponds to merging two sets
            # For example, to merge sets A and B, then merge the result with C, etc.
            # The number of operations is the number of merges, which is the number of steps minus one
            print(visited[current] - 1)
            return
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited[neighbor] = visited[current] + 1
                queue.append(neighbor)
    
    # If we exit the loop, it means M is not reachable from 1
    print(-1)

if __name__ == "__main__":
    main()