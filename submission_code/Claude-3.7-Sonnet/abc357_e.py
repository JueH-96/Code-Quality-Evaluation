def solve():
    n = int(input())
    edges = list(map(int, input().split()))
    
    # Count reachable pairs
    count = 0
    
    for u in range(1, n + 1):
        visited = [False] * (n + 1)
        current = u
        while not visited[current]:
            visited[current] = True
            count += 1
            current = edges[current - 1]
    
    return count

print(solve())