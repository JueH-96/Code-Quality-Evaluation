def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    path = []
    visited_in_path = set()
    current_vertex = 1
    
    while True:
        if current_vertex in visited_in_path:
            start_index = -1
            for i in range(len(path)):
                if path[i] == current_vertex:
                    start_index = i
                    break
            cycle_vertices = path[start_index:]
            print(len(cycle_vertices))
            print(*(cycle_vertices))
            return
        else:
            path.append(current_vertex)
            visited_in_path.add(current_vertex)
            current_vertex = a[current_vertex - 1]

if __name__ == '__main__':
    solve()