def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = input_data[1:]
    
    # Adjacency list: for each node, store list of (neighbor, cost)
    adj = [[] for _ in range(N)]
    sum_of_costs = 0
    
    idx = 0
    for _ in range(N - 1):
        A = int(edges[idx]) - 1
        B = int(edges[idx + 1]) - 1
        C = int(edges[idx + 2])
        idx += 3
        adj[A].append((B, C))
        adj[B].append((A, C))
        sum_of_costs += C
    
    # Function to find the farthest node and its distance from 'start'
    # using an iterative DFS (since it's a tree, no cycles except via parent).
    def get_farthest(start):
        dist = [-1] * N
        dist[start] = 0
        stack = [start]
        
        while stack:
            cur = stack.pop()
            for nxt, cost in adj[cur]:
                if dist[nxt] == -1:
                    dist[nxt] = dist[cur] + cost
                    stack.append(nxt)
        
        # Find node with maximum distance
        farthest_node = 0
        max_dist = 0
        for i in range(N):
            if dist[i] > max_dist:
                max_dist = dist[i]
                farthest_node = i
        return farthest_node, max_dist
    
    # 1) Pick an arbitrary node (e.g. 0), find the farthest node v
    v, _ = get_farthest(0)
    # 2) From v, find the farthest node w and its distance (this distance is the diameter)
    w, diameter = get_farthest(v)
    
    # The result: 2 * sum_of_costs - diameter
    answer = 2 * sum_of_costs - diameter
    print(answer)

# Call main to execute
if __name__ == "__main__":
    main()