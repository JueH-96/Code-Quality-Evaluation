from collections import deque, defaultdict
import sys

def bfs(graph, start):
    distances = {start: 0}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        
        for neighbor in graph[current]:
            if neighbor not in distances:
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)
    
    return distances

def min_operations_to_swap(N, M, S, T, edges):
    graph = defaultdict(list)
    
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Get distances from S and T
    dist_from_S = bfs(graph, S)
    dist_from_T = bfs(graph, T)
    
    # If A starts at S and B starts at T, we want A to reach T and B to reach S
    # A needs to move from S to T and B needs to move from T to S
    # We need to ensure they never meet at the same vertex during their moves
    
    # If A is at S and B is at T, we need to check the distances
    if T in dist_from_S and S in dist_from_T:
        distance_A_to_T = dist_from_S[T]
        distance_B_to_S = dist_from_T[S]
        
        # Check if they can reach their destinations without meeting
        for i in range(max(distance_A_to_T, distance_B_to_S) + 1):
            if (i <= distance_A_to_T and 
                (distance_A_to_T - i) <= distance_B_to_S and 
                (distance_B_to_S - (distance_A_to_T - i)) >= 0):
                return distance_A_to_T + distance_B_to_S
        
    return -1

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    first_line = list(map(int, data[0].split()))
    N, M, S, T = first_line[0], first_line[1], first_line[2], first_line[3]
    
    edges = []
    for i in range(1, M + 1):
        u, v = map(int, data[i].split())
        edges.append((u, v))
    
    result = min_operations_to_swap(N, M, S, T, edges)
    print(result)

if __name__ == "__main__":
    main()