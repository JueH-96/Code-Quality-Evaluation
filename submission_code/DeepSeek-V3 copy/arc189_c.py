from collections import defaultdict, deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))
    P = list(map(int, data[2+2*N:2+3*N]))
    Q = list(map(int, data[2+3*N:2+4*N]))
    
    # Adjust to 0-based index
    X -= 1
    P = [p-1 for p in P]
    Q = [q-1 for q in Q]
    
    # Create a graph where each node is a box, and edges represent the operation
    # We need to find a way to move all balls to X
    # We can model this as a graph where each operation moves balls from i to P[i] and Q[i]
    
    # We need to find the minimal number of operations to move all balls to X
    # We can think of this as a BFS problem where we try to move balls to X
    
    # First, find all boxes that have balls
    boxes_with_balls = set()
    for i in range(N):
        if A[i] > 0 or B[i] > 0:
            boxes_with_balls.add(i)
    
    if not boxes_with_balls:
        print(0)
        return
    
    # We need to find a way to move all balls to X
    # We can perform operations on boxes that have balls, and see if we can move them to X
    
    # We can model this as a BFS where each state is the set of boxes that have balls
    # But since N is up to 2e5, this is not feasible
    
    # Instead, we can model the problem as finding the minimal number of operations to move all balls to X
    # We can think of the problem as moving balls along the graph defined by P and Q
    
    # For each box that has balls, we need to find the minimal number of operations to move its balls to X
    
    # We can precompute the distance from each box to X in the graph defined by P and Q
    
    # Create a graph where each node has edges to P[i] and Q[i]
    graph = defaultdict(list)
    for i in range(N):
        graph[i].append(P[i])
        graph[i].append(Q[i])
    
    # Compute the distance from each node to X
    distance = [float('inf')] * N
    distance[X] = 0
    queue = deque([X])
    
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if distance[neighbor] == float('inf'):
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    
    # Now, for each box that has balls, we need to find the minimal number of operations to move its balls to X
    # The number of operations is the distance from that box to X
    
    total_operations = 0
    for box in boxes_with_balls:
        if distance[box] == float('inf'):
            print(-1)
            return
        total_operations += distance[box]
    
    print(total_operations)

if __name__ == "__main__":
    main()