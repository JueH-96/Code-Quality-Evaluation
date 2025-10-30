from collections import deque, defaultdict
import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    N, M, K = map(int, data[0].split())
    
    graph = defaultdict(list)
    
    # Read edges
    for i in range(1, M + 1):
        a, b = map(int, data[i].split())
        graph[a].append(b)
        graph[b].append(a)
    
    guards = []
    
    # Read guards
    for i in range(M + 1, M + K + 1):
        p, h = map(int, data[i].split())
        guards.append((p, h))
    
    # To keep track of guarded vertices
    guarded = set()
    
    # BFS for each guard
    for p, h in guards:
        queue = deque([(p, 0)])  # (current_vertex, current_distance)
        visited = set([p])
        
        while queue:
            current, distance = queue.popleft()
            if distance > h:
                continue
            
            guarded.add(current)
            
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
    
    # Prepare the output
    guarded_list = sorted(guarded)
    print(len(guarded_list))
    print(" ".join(map(str, guarded_list)))

if __name__ == "__main__":
    main()