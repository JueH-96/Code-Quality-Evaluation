def main():
    import sys
    from collections import deque
    
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges_data = input_data[2:]
    
    # We will have 2*N variables in total:
    #   T_i  for each villager i in [0..N-1]  (whether i's testimonies are true(1) or false(0))
    #   H_i  for each villager i in [0..N-1]  (whether villager i is honest(1) or liar(0))
    #
    # We store these 2*N variables as nodes in a graph:
    #   Node i     = T_i   (0 <= i < N)
    #   Node N + i = H_i   (N <= N+i < 2N)
    #
    # Each testimony "A -> B with w (0 or 1)" means:
    #   H_B  xor  T_A  =  w
    # i.e. "the honesty of B" XOR "whether A's statements are true" = w
    #
    # We will build these constraints as edges in a graph, each edge labeled with the XOR requirement.
    #
    # To find a solution, we do a BFS/DFS over this graph of size 2*N:
    #   - If a node is unvisited, assign it arbitrarily (0).
    #   - Propagate the assignments via the edges:
    #       if x xor y = label, then y = x xor label
    #   - If we ever get a conflict, no solution => print(-1).
    #   - Otherwise, we recover T_i and H_i, then confusion bit c_i = T_i xor H_i.

    adjacency = [[] for _ in range(2*N)]
    idx = 0
    for _ in range(M):
        A = int(edges_data[idx])   # villager A
        B = int(edges_data[idx+1]) # villager B
        w = int(edges_data[idx+2]) # 0 => "B is honest", 1 => "B is liar"
        idx += 3
        # Convert to zero-based
        A -= 1
        B -= 1
        # T_A is node A, H_B is node (N+B).
        adjacency[A].append((N+B, w))
        adjacency[N+B].append((A, w))
    
    # val[u] will hold the assigned value (0 or 1) for node u, or -1 if unassigned
    val = [-1]*(2*N)
    
    # We'll do BFS over each connected component to assign values consistently
    for start in range(2*N):
        if val[start] == -1:
            val[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for (v, xorlbl) in adjacency[u]:
                    if val[v] == -1:
                        # assign v so that val[u] xor val[v] = xorlbl
                        val[v] = val[u] ^ xorlbl
                        queue.append(v)
                    else:
                        # check consistency
                        if val[v] != (val[u] ^ xorlbl):
                            print(-1)
                            return
    
    # If we are here, we have a consistent assignment.
    # T_i = val[i], H_i = val[N + i].
    # Confusion bit c_i = T_i xor H_i
    #   (because T_i = 1 means i's statements are true => i is either honest+not_confused or liar+confused
    #    T_i = 0 means i's statements are false => i is either liar+not_confused or honest+confused)
    
    answer = []
    for i in range(N):
        c_i = val[i] ^ val[N+i]  # T_i xor H_i
        answer.append(str(c_i))
    
    print("".join(answer))

# Do not forget to call main().
if __name__ == "__main__":
    main()