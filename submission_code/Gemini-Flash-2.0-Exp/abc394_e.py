def solve():
    n = int(input())
    graph = []
    for _ in range(n):
        graph.append(input())

    def find_shortest_palindrome_path(start, end):
        q = [(start, end, 0)]
        visited = set()
        visited.add((start, end))

        while q:
            u, v, dist = q.pop(0)

            if u == v:
                return dist
            
            # Expand from u
            for neighbor_u in range(n):
                if graph[u][neighbor_u] != '-':
                    # Expand from v
                    for neighbor_v in range(n):
                        if graph[v][neighbor_v] != '-':
                            if graph[u][neighbor_u] == graph[v][neighbor_v]:
                                if (neighbor_u, neighbor_v) not in visited:
                                    q.append((neighbor_u, neighbor_v, dist + 1))
                                    visited.add((neighbor_u, neighbor_v))
            
            # Special case: u == end
            if u == end:
                return dist
            
        return -1

    result = []
    for i in range(n):
        row_result = []
        for j in range(n):
            row_result.append(str(find_shortest_palindrome_path(i, j)))
        result.append(" ".join(row_result))

    for row in result:
        print(row)

solve()