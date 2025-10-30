# YOUR CODE HERE
import sys
import collections

def main():
    """
    This function contains the main logic of the solution.
    It's encapsulated in a function to avoid global scope issues
    and to be more organized.
    """
    # Fast I/O
    input = sys.stdin.readline

    # Read input
    try:
        N, M, S, T = map(int, input().split())
    except ValueError:
        # Handles empty input line
        return
        
    S -= 1
    T -= 1
    
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)

    # The state of the system is (position of A, position of B).
    # We want to find the shortest path from (S, T) to (T, S) in the state graph.
    # The number of states is N*(N-1), which can be large.
    # We use a bidirectional BFS to find the shortest path efficiently.

    # Forward search from (S, T)
    q_f = collections.deque([(S, T)])
    dist_f = {(S, T): 0}
    
    # Backward search from (T, S)
    q_b = collections.deque([(T, S)])
    dist_b = {(T, S): 0}
    
    ans = float('inf')

    # If the start and end states are the same for one of the searches,
    # and that state is met by the other search, the path length is d_f + d_b.
    if (S, T) in dist_b:
        ans = min(ans, dist_f[(S,T)] + dist_b[(S,T)])

    # We process one level of the search at a time.
    # In each iteration of the while loop, we expand one level from both searches.
    while q_f and q_b and ans == float('inf'):
        
        # Expand forward search. To ensure we only expand one level,
        # we iterate over the current size of the queue.
        len_q_f = len(q_f)
        for _ in range(len_q_f):
            u_A, u_B = q_f.popleft()
            d = dist_f[(u_A, u_B)]

            # Try moving piece A
            for v_A in adj[u_A]:
                if v_A == u_B:  # Collision check
                    continue
                if (v_A, u_B) not in dist_f:
                    dist_f[(v_A, u_B)] = d + 1
                    q_f.append((v_A, u_B))
                    if (v_A, u_B) in dist_b:
                        ans = min(ans, d + 1 + dist_b[(v_A, u_B)])
            
            # Try moving piece B
            for v_B in adj[u_B]:
                if v_B == u_A:  # Collision check
                    continue
                if (u_A, v_B) not in dist_f:
                    dist_f[(u_A, v_B)] = d + 1
                    q_f.append((u_A, v_B))
                    if (u_A, v_B) in dist_b:
                        ans = min(ans, d + 1 + dist_b[(u_A, v_B)])
        
        if ans != float('inf'):
            break

        # Expand backward search
        len_q_b = len(q_b)
        for _ in range(len_q_b):
            u_A, u_B = q_b.popleft()
            d = dist_b[(u_A, u_B)]
            
            # Try moving piece A (from the perspective of the target configuration)
            for v_A in adj[u_A]:
                if v_A == u_B:
                    continue
                if (v_A, u_B) not in dist_b:
                    dist_b[(v_A, u_B)] = d + 1
                    q_b.append((v_A, u_B))
                    if (v_A, u_B) in dist_f:
                        ans = min(ans, d + 1 + dist_f[(v_A, u_B)])
                        
            # Try moving piece B
            for v_B in adj[u_B]:
                if v_B == u_A:
                    continue
                if (u_A, v_B) not in dist_b:
                    dist_b[(u_A, v_B)] = d + 1
                    q_b.append((u_A, v_B))
                    if (u_A, v_B) in dist_f:
                        ans = min(ans, d + 1 + dist_f[(u_A, v_B)])

    if ans == float('inf'):
        print(-1)
    else:
        print(ans)

# Enclose the call to main in a standard boilerplate for Python scripts
if __name__ == '__main__':
    main()