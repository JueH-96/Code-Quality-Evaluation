import sys

def solve():
    n, m, q = map(int, sys.stdin.readline().split())
    roads = []
    for _ in range(m):
        u, v, w = map(int, sys.stdin.readline().split())
        roads.append(((u, v), w))
    
    is_road_open = [True] * m
    
    def get_shortest_paths(current_open_roads):
        dist_matrix = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist_matrix[i][i] = 0
        
        for i in range(m):
            if current_open_roads[i]:
                (u, v), weight = roads[i]
                dist_matrix[u-1][v-1] = min(dist_matrix[u-1][v-1], weight)
                dist_matrix[v-1][u-1] = min(dist_matrix[v-1][u-1], weight)
                
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist_matrix[i][k] != float('inf') and dist_matrix[k][j] != float('inf'):
                        dist_matrix[i][j] = min(dist_matrix[i][j], dist_matrix[i][k] + dist_matrix[k][j])
                        
        return dist_matrix
        
    shortest_distances = get_shortest_paths(is_road_open)
    
    for _ in range(q):
        query = list(map(int, sys.stdin.readline().split()))
        if query[0] == 1:
            road_index_to_close = query[1]
            is_road_open[road_index_to_close-1] = False
            shortest_distances = get_shortest_paths(is_road_open)
        elif query[0] == 2:
            start_city = query[1]
            end_city = query[2]
            distance = shortest_distances[start_city-1][end_city-1]
            if distance == float('inf'):
                print("-1")
            else:
                print(distance)

if __name__ == '__main__':
    solve()