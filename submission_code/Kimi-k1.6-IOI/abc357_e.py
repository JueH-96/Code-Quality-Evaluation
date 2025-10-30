def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    a = list(map(int, data[1:N+1]))
    
    visited = [False] * (N + 1)
    answer = [0] * (N + 1)
    
    for u in range(1, N + 1):
        if not visited[u]:
            path = []
            pos = dict()
            current = u
            while True:
                if visited[current]:
                    if current in pos:
                        # Found a cycle
                        c = pos[current]
                        L = len(path) - c
                        # Process cycle nodes
                        for i in range(c, len(path)):
                            node = path[i]
                            answer[node] = L
                            visited[node] = True
                        # Process nodes before cycle
                        for i in range(c):
                            node = path[i]
                            answer[node] = (c - i) + L
                            visited[node] = True
                    else:
                        # Process tree nodes
                        k = len(path)
                        ans_current = answer[current]
                        for i in range(k):
                            node = path[i]
                            answer[node] = (k - i) + ans_current
                            visited[node] = True
                    break
                # Add current to path
                pos[current] = len(path)
                path.append(current)
                current = a[current - 1]  # convert to 0-based index
    
    total = sum(answer)
    print(total)

if __name__ == "__main__":
    main()