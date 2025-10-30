def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1+n]))
    
    graph = [0] * (n+1)
    for i in range(1, n+1):
        graph[i] = arr[i-1]
    
    visited_global = [False] * (n+1)
    for i in range(1, n+1):
        if visited_global[i]:
            continue
        path = []
        path_map = {}
        cur = i
        while True:
            if cur in path_map:
                start_index = path_map[cur]
                cycle = path[start_index:]
                print(len(cycle))
                print(" ".join(map(str, cycle)))
                return
                
            if visited_global[cur]:
                break
                
            visited_global[cur] = True
            path.append(cur)
            path_map[cur] = len(path) - 1
            
            cur = graph[cur]

if __name__ == "__main__":
    main()