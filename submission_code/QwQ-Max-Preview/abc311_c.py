def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    visited = [False] * (N + 1)
    
    for i in range(1, N + 1):
        if not visited[i]:
            path = []
            path_indices = {}
            current = i
            while True:
                if current in path_indices:
                    start = path_indices[current]
                    cycle = path[start:]
                    print(len(cycle))
                    print(' '.join(map(str, cycle)))
                    return
                if visited[current]:
                    break
                visited[current] = True
                path_indices[current] = len(path)
                path.append(current)
                current = A[current - 1]  # A is 0-based in input

if __name__ == "__main__":
    main()