def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    visited = [False] * (N + 1)
    
    for vertex in range(1, N + 1):
        if not visited[vertex]:
            path = []
            pos = {}
            current = vertex
            while True:
                if visited[current]:
                    if current in pos:
                        # Extract the cycle from pos[current] to the end, excluding the last vertex
                        cycle = path[pos[current]:]
                        print(len(cycle))
                        print(' '.join(map(str, cycle)))
                        return
                    else:
                        break
                visited[current] = True
                pos[current] = len(path)
                path.append(current)
                current = A[current - 1]

if __name__ == '__main__':
    main()