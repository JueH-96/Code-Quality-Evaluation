def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    
    adj = [set() for _ in range(N + 1)]
    degrees = [0] * (N + 1)
    
    for _ in range(N - 1):
        u = int(input[idx])
        v = int(input[idx + 1])
        idx += 2
        adj[u].add(v)
        adj[v].add(u)
        degrees[u] += 1
        degrees[v] += 1
    
    S = [u for u in range(1, N + 1) if degrees[u] == 2]
    valid_pairs = set()
    
    for u in S:
        for v in adj[u]:
            if degrees[v] != 3:
                continue
            stack = [(v, u)]
            visited = set()
            while stack:
                current, parent = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                for neighbor in adj[current]:
                    if neighbor == parent:
                        continue
                    if degrees[neighbor] == 3:
                        stack.append((neighbor, current))
                    elif degrees[neighbor] == 2:
                        if neighbor not in adj[u]:
                            a, b = (u, neighbor) if u < neighbor else (neighbor, u)
                            valid_pairs.add((a, b))
    
    print(len(valid_pairs))

if __name__ == "__main__":
    main()