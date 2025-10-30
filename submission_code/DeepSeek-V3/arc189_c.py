from collections import deque

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
    
    # Initialize the total number of balls
    total_red = sum(A)
    total_blue = sum(B)
    
    # If no balls, check if X is already empty
    if total_red == 0 and total_blue == 0:
        # Check if all boxes except X are empty
        all_empty = True
        for i in range(N):
            if i != X and (A[i] != 0 or B[i] != 0):
                all_empty = False
                break
        if all_empty:
            print(0)
            return
        else:
            print(-1)
            return
    
    # We need to find a way to move all balls to X
    # We can model this as a graph where each node represents a box
    # and edges represent the possible moves
    
    # We need to find a sequence of operations that moves all balls to X
    # The operations are: choose a box, move its red balls to P_i and blue balls to Q_i
    
    # To find the minimum number of operations, we can perform a BFS-like approach
    # where we track the state of the boxes and the number of operations
    
    # However, since N is up to 2e5, we need a smarter approach
    
    # Instead of tracking the state of all boxes, we can track the positions of the balls
    # and find the minimum number of operations to move all balls to X
    
    # Since each operation moves all balls from a box to P_i and Q_i, we can think of it as
    # moving the balls along the edges of the graph
    
    # We need to find the shortest path for each ball to reach X
    
    # For each ball, we can perform a BFS to find the minimum number of operations to reach X
    
    # Since the balls are independent, we can compute the maximum number of operations needed
    # for any ball to reach X
    
    # So, for each ball, we find the minimum number of operations to reach X, and the answer is
    # the maximum of these values
    
    # Now, we need to find the minimum number of operations for each ball to reach X
    
    # We can precompute the distance from each box to X using BFS
    
    # First, build the graph for red balls
    # For red balls, moving from i to P_i
    # So, the graph is i -> P_i
    
    # Similarly, for blue balls, moving from i to Q_i
    # So, the graph is i -> Q_i
    
    # We need to find the distance from each box to X in both graphs
    
    # Compute distance for red balls
    red_distance = [-1] * N
    queue = deque()
    queue.append(X)
    red_distance[X] = 0
    
    while queue:
        u = queue.popleft()
        for v in [P[u]]:
            if red_distance[v] == -1:
                red_distance[v] = red_distance[u] + 1
                queue.append(v)
    
    # Compute distance for blue balls
    blue_distance = [-1] * N
    queue = deque()
    queue.append(X)
    blue_distance[X] = 0
    
    while queue:
        u = queue.popleft()
        for v in [Q[u]]:
            if blue_distance[v] == -1:
                blue_distance[v] = blue_distance[u] + 1
                queue.append(v)
    
    # Now, for each ball, find the minimum number of operations to reach X
    max_ops = 0
    for i in range(N):
        if A[i] == 1:
            if red_distance[i] == -1:
                print(-1)
                return
            max_ops = max(max_ops, red_distance[i])
        if B[i] == 1:
            if blue_distance[i] == -1:
                print(-1)
                return
            max_ops = max(max_ops, blue_distance[i])
    
    print(max_ops)

if __name__ == "__main__":
    main()