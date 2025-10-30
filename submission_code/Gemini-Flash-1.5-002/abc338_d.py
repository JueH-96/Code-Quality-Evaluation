# YOUR CODE HERE
def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))

    def get_tour_length(closed_bridge, tour):
        adj = [[] for _ in range(n + 1)]
        for i in range(1, n):
            adj[i].append(i + 1)
            adj[i + 1].append(i)
        adj[n].append(1)
        adj[1].append(n)
        
        if closed_bridge != n:
            adj[closed_bridge].remove(closed_bridge + 1)
            adj[closed_bridge + 1].remove(closed_bridge)
        else:
            adj[n].remove(1)
            adj[1].remove(n)

        
        q = [(tour[0], 0, [tour[0]])]
        min_len = float('inf')

        while q:
            curr, length, path = q.pop(0)
            
            if len(path) == len(tour) and path[-1] == tour[-1]:
                min_len = min(min_len, length)
                continue

            for neighbor in adj[curr]:
                new_path = path + [neighbor]
                
                valid_path = True
                
                
                path_idx = 0
                tour_idx = 0
                
                while tour_idx < len(tour) and path_idx < len(new_path):
                    if new_path[path_idx] == tour[tour_idx]:
                        tour_idx += 1
                    path_idx += 1
                
                if tour_idx != len(tour):
                    valid_path = False
                
                if valid_path:
                    q.append((neighbor, length + 1, new_path))

        return min_len

    min_total_length = float('inf')
    for closed_bridge in range(1, n + 1):
        min_total_length = min(min_total_length, get_tour_length(closed_bridge, x))

    print(min_total_length)

solve()