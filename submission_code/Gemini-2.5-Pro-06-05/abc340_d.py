import sys
import heapq

def solve():
    """
    This function solves the problem by modeling it as a shortest path problem
    on a graph and using Dijkstra's algorithm.
    """
    # Use fast I/O for performance with large inputs.
    input = sys.stdin.readline
    
    # Read the total number of stages.
    N = int(input())
    
    # Read the action parameters for each stage from 1 to N-1.
    # actions[i] stores the tuple (A_{i+1}, B_{i+1}, X_{i+1}).
    # This means for stage `u`, the parameters are at `actions[u-1]`.
    actions = [list(map(int, input().split())) for _ in range(N - 1)]

    # `dist[i]` stores the minimum time to unlock stage `i`. We use 1-based indexing.
    # Initialize all distances to infinity.
    dist = [float('inf')] * (N + 1)
    
    # The time to unlock the starting stage 1 is 0.
    dist[1] = 0
    
    # Priority queue for Dijkstra's algorithm. It stores tuples of (time, stage).
    # It is a min-heap, always providing the unvisited node with the smallest time.
    pq = [(0, 1)]
    
    # Main loop of Dijkstra's algorithm.
    while pq:
        # Pop the stage with the smallest current time from the priority queue.
        time, u = heapq.heappop(pq)
        
        # If the popped time is greater than the already known shortest time to u,
        # it is a stale entry from a previously longer path, so we ignore it.
        if time > dist[u]:
            continue
        
        # Optimization: If we have reached the destination stage N, we can stop.
        # In Dijkstra's, the first time we extract the destination, its shortest path is finalized.
        if u == N:
            break
            
        # Get the action parameters for the current stage `u`.
        # These are stored at index `u-1` in our `actions` list.
        a_u, b_u, x_u = actions[u - 1]
        
        # Explore the first action: clear stage u to unlock u+1.
        # This corresponds to an edge from u to v1 = u+1 with weight a_u.
        v1 = u + 1
        if time + a_u < dist[v1]:
            dist[v1] = time + a_u
            heapq.heappush(pq, (dist[v1], v1))
            
        # Explore the second action: clear stage u to unlock X_u.
        # This corresponds to an edge from u to v2 = x_u with weight b_u.
        v2 = x_u
        if time + b_u < dist[v2]:
            dist[v2] = time + b_u
            heapq.heappush(pq, (dist[v2], v2))
            
    # The final answer is the minimum time to unlock stage N.
    print(dist[N])

if __name__ == "__main__":
    solve()